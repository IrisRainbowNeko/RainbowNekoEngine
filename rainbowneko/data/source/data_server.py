import atexit
import multiprocessing as mp
import pickle
import platform
import time
import weakref
from contextlib import contextmanager

import zmq
from torch.utils.data import get_worker_info

from rainbowneko import _share
from .base import DataSource, ComposeWebdsSource


def cleanup_worker(process):
    if process and process.is_alive():
        try:
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
        except Exception:
            pass


def data_load_worker(source: ComposeWebdsSource, port, stop_event):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.set_hwm(1000)

    try:
        socket.bind(f"tcp://*:{port}")
    except zmq.ZMQError:
        return

    try:
        while not stop_event.is_set():
            with source.return_source():
                for data, source_i in source:
                    if stop_event.is_set():
                        break

                    sidx = source.source_list.index(source_i)
                    try:
                        socket.send(pickle.dumps((data, sidx)), copy=False)
                    except zmq.ZMQError:
                        break
    except Exception:
        pass
    finally:
        socket.close()
        context.term()


class DataServerSource(DataSource):
    def __init__(self, source: ComposeWebdsSource, port=29300):
        self.source = source
        self._return_source = False
        self.server_process = None

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        local_rank = _share.local_rank

        current_port = port + local_rank

        if worker_id == 0 and local_rank >= 0:
            ctx = self.get_context()
            self.stop_event = ctx.Event()

            p = ctx.Process(
                target=data_load_worker,
                args=(source, current_port, self.stop_event)
            )
            p.daemon = True
            p.start()

            self.server_process = p
            self._finalizer = weakref.finalize(self, cleanup_worker, p)
            atexit.register(cleanup_worker, p)

        if worker_id == 0:
            time.sleep(0.2)

        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://127.0.0.1:{current_port}")
        self.socket = socket

    @contextmanager
    def return_source(self):
        self._return_source = True
        yield
        self._return_source = False

    def get_context(self):
        if platform.system() == "Linux":
            return mp.get_context('fork')
        else:
            return mp.get_context('spawn')

    def get_data(self):
        try:
            if self.socket.poll(timeout=600000):
                data_bytes = self.socket.recv()
                return pickle.loads(data_bytes)
            return None, None
        except zmq.ZMQError:
            return None, None

    def __iter__(self):
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
        if hasattr(self, 'socket'):
            self.socket.close()