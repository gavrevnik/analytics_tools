import os
import pandas as pd
import numpy as np
import _thread
import time
import random
import threading
import queue
import multiprocessing
import joblib


def fork_ex1():
    # идентификация процессов
    print(f'{os.getpid()} - start')
    newpid = os.fork()
    # разветвляемся
    if newpid == 0:
        x = 'child'
    else:
        x = 'main'
    print(f'Calc from {os.getpid()} x = {x}')

def fork_ex2():
    # независимая запись процессов в файлы
    x = [1,2,3]
    newpid = os.fork()
    if newpid == 0:
        # child
        x.append(1)
        pd.DataFrame({'x' : x}).to_csv('./child.csv')
    else:
        x.append(7)
        pd.DataFrame({'x' : x}).to_csv('./parent.csv')

def thread_ex1():
    """Многопоточный расчет на смежных данных"""
    class Storage:
        def __init__(self, x, mutex):
            self.x = x
            self.mutex = mutex # блокировка доступа
        def thread_func(self, tid, sleep=0):

            print(f'child thread {tid} prepared')
            mutex.acquire() # lock
            print(f'child thread {tid} started x = {self.x}')
            time.sleep(sleep)
            self.x = self.x + 1
            print(f'child thread {tid} result = {self.x}')
            mutex.release() # unlock

    mutex = _thread.allocate_lock()
    ins = Storage(5, mutex)
    _thread.start_new_thread(ins.thread_func, (1,))
    _thread.start_new_thread(ins.thread_func, (2,))
    ins.thread_func(0, )
    time.sleep(5)

def thread_ex2():
    """Последовательный счет случайными потоками с блокировкой вывода"""
    stdoutmutex = _thread.allocate_lock()
    thread_finish_list = [False] * 5
    global g_count
    g_count = 0
    def counter(myId, count):
        for i in range(count):
            with stdoutmutex: # stdoutmutex.acquire() ... stdoutmutex.release()
                global g_count
                g_count = g_count + 1
                print('[%s] => %s' % (myId, g_count))


        thread_finish_list[myId] = True
    for i in range(5):
        _thread.start_new_thread(counter, (i, 5))
    while False in thread_finish_list:
        pass
    print('finish')

def threading_ex1():
    """Аналогичная задача через модуль threading"""
    class MyThread(threading.Thread):
        def __init__(self, tid, count, mutex):
            self.tid = tid
            self.count = count
            self.mutex = mutex
            threading.Thread.__init__(self)
        # override Thread.run()
        def run(self):
            for j in range(self.count):
                with self.mutex:
                    print(f'{tid}->{j}')
    count = 5
    mutex = threading.Lock()
    threads = []
    # создаем 5 потоков
    for tid in range(5):
        thread = MyThread(tid, count, mutex) # thread init
        thread.start()
        threads.append(thread)
    # пауза пока поток не завершится
    for th in threads:
        th.join()
    print('finish')

def threading_ex2():
    """Передача функций в многопоточный класс. Параллельный счет суммы"""
    global fin_list
    fin_list = []
    N_th = 5
    mutex = threading.Lock()
    def func(tid, start, mutex, step):
        global fin_list
        tmp_val = val_list[start:start+step]
        tmp_val = sum(tmp_val)
        with mutex:
            print(f'{tid} started')
            fin_list.append(tmp_val)
            print(f'{tid} done {tmp_val}')
    start = 0
    val_list = np.random.choice(list(range(30)), 30, replace=True)
    threads = []
    print(val_list)
    step = int(len(val_list)/N_th)
    for tid in range(N_th):
        thread = threading.Thread(target=func, args=(tid, start, mutex, step))
        thread.start()
        start = start + step
        threads.append(thread)

    for th in threads:
        th.join()
    print('finish')
    print(fin_list)
    print(sum(val_list), sum(fin_list))

def threading_ex3():
    numconsumers, numproducers, nummessages = 1, 5, 5
    printmutex = threading.Lock()

    q = queue.Queue()
    def producer(printmutex, tid, nummessages):
        for msg in range(nummessages):
            q.put(msg)
            with printmutex:
                print(f'producer {tid} put {msg}')

    def consumer(printmutex, tid, nummessages):
        for _ in range(nummessages):
            try:
                msg = q.get(block=False)
            except:
                pass
            with printmutex:
                print(f'consumer {tid} get {msg}')

    threads = []
    for tid in range(numproducers):
        thread = threading.Thread(target=producer, args=(printmutex, tid, nummessages))
        # threads.append(thread)
        thread.start()

    for tid in range(numconsumers):
        thread = threading.Thread(target=consumer, args=(printmutex, tid, nummessages))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def threading_ex4():
    def server():
        while True:
            k = input()
            print(f'hello {k}')
            if k == 'q':
                break
    print('init')
    t=threading.Thread(target=server)
    t.daemon=True # terminate when parent thread terminated
    t.start()
    t.join() # wait from this thread until t terminate
    print('parent process exiting')

def multiprocessing_ex1():
    mutex = multiprocessing.Lock()
    global x
    x = []
    def whoami(name, mutex):
        with mutex:
            print(f'process {name} pid {os.getpid()}')
            x.append(random.randint(0,10))
            print(x)
    whoami('parent', mutex)
    for p in range(5):
        pr = multiprocessing.Process(target=whoami, args=(p, mutex))
        pr.start()
        pr.join() # wait until process end

    with mutex:
        print('parent finish')

# def work_func(x):
#     print(f'worker {os.getpid()}')
#     return x**2
# workersnum = 5
# raw_list = list(range(10))
# print(raw_list)
# workers=multiprocessing.Pool(processes=workersnum)
# results = workers.map(work_func, raw_list)
# print(results)


