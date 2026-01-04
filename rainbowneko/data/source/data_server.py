import atexit
import threading
import pickle
import time
import weakref
from contextlib import contextmanager

import zmq
from torch.utils.data import get_worker_info

from rainbowneko import _share
from .base import DataSource, ComposeWebdsSource


def cleanup_worker(thread, stop_event):
    """Thread cleanup function"""
    if thread and thread.is_alive():
        stop_event.set()  # Notify thread to stop
        thread.join(timeout=2)


def data_load_worker(source: ComposeWebdsSource, port, num_workers, stop_event):
    context = zmq.Context()

    # Data transmission port (PUSH)
    data_socket = context.socket(zmq.PUSH)
    data_socket.setsockopt(zmq.LINGER, 0)
    data_socket.set_hwm(1000)

    # Signal receiving port (REP), suggest port + 1 to avoid conflicts
    sync_port = port + 1

    try:
        data_socket.bind(f"tcp://*:{port}")
    except zmq.ZMQError as e:
        print(f"DataServer Bind Error (Data): {e}")
        return

    try:
        while not stop_event.is_set():
            # --- Wait for Client signal ---
            sync_socket = context.socket(zmq.REP)
            sync_socket.setsockopt(zmq.LINGER, 0)
            try:
                sync_socket.bind(f"tcp://*:{sync_port}")
                # Block until any message is received
                _ = sync_socket.recv()
                sync_socket.send(b"OK")  # Reply with confirmation signal
            except zmq.ZMQError:
                continue
            finally:
                sync_socket.close()
            # ---------------------------

            if stop_event.is_set():
                break

            with source.return_source():
                for i, (data, source_i) in enumerate(source):
                    if stop_event.is_set():
                        break

                    sidx = source.source_list.index(source_i)
                    try:
                        data_socket.send(pickle.dumps((data, sidx)), copy=False)
                    except zmq.ZMQError:
                        break

                # Send termination signal to each worker
                for _ in range(num_workers):
                    try:
                        data_socket.send(pickle.dumps((None, None)), copy=False)
                    except zmq.ZMQError:
                        break

    except Exception as e:
        print(f"DataServer Error: {e}")
    finally:
        data_socket.close()
        context.term()


class DataServerSource(DataSource):
    def __init__(self, source: ComposeWebdsSource, port=29300):
        self.source = source
        self._return_source = False
        self.server_thread = None
        self.port = port
        self.socket = None
        self.stop_event = None

    @contextmanager
    def return_source(self):
        self._return_source = True
        yield
        self._return_source = False

    def get_data(self):
        try:
            if self.socket.poll(timeout=10000):
                data_bytes = self.socket.recv()
                return pickle.loads(data_bytes)
            return None, None
        except zmq.ZMQError:
            return None, None

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        local_rank = getattr(_share, 'local_rank', 0)

        # 1. Start Server thread (only executed by the first Worker of Rank 0)
        if self.socket is None:
            if worker_id == 0 and local_rank <= 0:
                self.stop_event = threading.Event()
                t = threading.Thread(
                    target=data_load_worker,
                    args=(self.source, self.port, num_workers, self.stop_event),
                    name="DataServerThread"
                )
                t.daemon = True
                t.start()
                self.server_thread = t
                self._finalizer = weakref.finalize(self, cleanup_worker, t, self.stop_event)
                atexit.register(cleanup_worker, t, self.stop_event)

                # Give the Server a small amount of time to bind
                time.sleep(0.2)

            # 2. Establish data receiving connection
            context = zmq.Context()
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://127.0.0.1:{self.port}")
            self.socket = socket

        # 3. Send "start signal" to Server (ensure Server begins iterating data)
        # To prevent race conditions from all workers sending, usually triggered by worker_id 0
        if worker_id == 0 and local_rank <= 0:
            sync_context = zmq.Context()
            sync_sock = sync_context.socket(zmq.REQ)
            sync_sock.setsockopt(zmq.LINGER, 0)
            sync_sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout to prevent deadlock
            sync_sock.connect(f"tcp://127.0.0.1:{self.port + 1}")
            try:
                sync_sock.send(b"START")
                sync_sock.recv()  # Wait for Server's "OK"
            except zmq.ZMQError:
                print("DataServer Sync Timeout/Error. Data might not start.")
            finally:
                sync_sock.close()
                sync_context.term()

        return self

    def __next__(self):
        data_tuple = self.get_data()

        if data_tuple == (None, None):
            raise StopIteration

        data, sidx = data_tuple
        if self._return_source:
            return data, self.source.source_list[sidx]
        else:
            return data

    def __del__(self):
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()

    def __getitem__(self, index):
        raise NotImplementedError('DataServerSource can only be applied to stream source.')

    def __len__(self):
        return len(self.source)