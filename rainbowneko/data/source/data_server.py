import atexit
import threading  # 替换 multiprocessing
import pickle
import time
import weakref
from contextlib import contextmanager

import zmq
from torch.utils.data import get_worker_info

from rainbowneko import _share
from .base import DataSource, ComposeWebdsSource


def cleanup_worker(thread, stop_event):
    """线程清理函数"""
    if thread and thread.is_alive():
        stop_event.set()  # 通知线程停止
        thread.join(timeout=2)


def data_load_worker(source: ComposeWebdsSource, port, num_workers, stop_event):
    context = zmq.Context()

    # 数据发送端口 (PUSH)
    data_socket = context.socket(zmq.PUSH)
    data_socket.setsockopt(zmq.LINGER, 0)
    data_socket.set_hwm(1000)

    # 信号接收端口 (REP)，建议使用 port + 1 避免冲突
    sync_port = port + 1

    try:
        data_socket.bind(f"tcp://*:{port}")
    except zmq.ZMQError as e:
        print(f"DataServer Bind Error (Data): {e}")
        return

    try:
        while not stop_event.is_set():
            # --- 新增：等待 Client 信号 ---
            sync_socket = context.socket(zmq.REP)
            sync_socket.setsockopt(zmq.LINGER, 0)
            try:
                sync_socket.bind(f"tcp://*:{sync_port}")
                # 阻塞直到收到任何消息
                _ = sync_socket.recv()
                sync_socket.send(b"OK")  # 回复确认信号
            except zmq.ZMQError:
                continue
            finally:
                sync_socket.close()
            # ---------------------------

            if stop_event.is_set():
                break

            with source.return_source():
                for i,(data, source_i) in enumerate(source):
                    if stop_event.is_set():
                        break

                    sidx = source.source_list.index(source_i)
                    try:
                        data_socket.send(pickle.dumps((data, sidx)), copy=False)
                    except zmq.ZMQError:
                        break

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
            if self.socket.poll(timeout=600000):
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

        # 1. 启动 Server 线程 (仅由 Rank 0 的第一个 Worker 执行)
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

                # 给 Server 启动 bind 留出微小时间
                time.sleep(0.2)

            # 2. 发送“启动信号”给 Server (确保 Server 开始迭代数据)
            # 为了防止所有 worker 同时发送导致竞争，通常由 worker_id 0 触发即可
            if worker_id == 0 and local_rank <= 0:
                sync_context = zmq.Context()
                sync_sock = sync_context.socket(zmq.REQ)
                sync_sock.setsockopt(zmq.LINGER, 0)
                sync_sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时防止死锁
                sync_sock.connect(f"tcp://127.0.0.1:{self.port + 1}")
                try:
                    sync_sock.send(b"START")
                    sync_sock.recv()  # 等待 Server 的 "OK"
                except zmq.ZMQError:
                    print("DataServer Sync Timeout/Error. Data might not start.")
                finally:
                    sync_sock.close()
                    sync_context.term()

            # 3. 建立数据接收连接
            context = zmq.Context()
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://127.0.0.1:{self.port}")
            self.socket = socket

        return self

    def __next__(self):
        data_tuple = self.get_data()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if data_tuple == (None, None):
            print(worker_id)
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