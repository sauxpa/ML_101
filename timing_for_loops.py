#!/usr/bin/env python
# coding: utf-8

import itertools
import timeit
import numpy as np


n = 1000


def fizz_buzz():
    for i in range(1, n + 1):
        fizz = 'Fizz' if i % 3 == 0 else ''  # noqa F841
        buzz = 'Buzz' if i % 5 == 0 else ''  # noqa F841


def fizz_buzz_list_comprehension():
    [
        'FizzBuzz' if i % 3 == 0 and i % 5 == 0 else (
            'Fizz' if i % 3 == 0 else (
                'Buzz' if i % 5 == 0 else (
                    i
                )
            )
        )
        for i in range(1, n + 1)
    ]


def fizz_buzz_itertools():
    fizzes = itertools.cycle([""] * 2 + ["Fizz"])
    buzzes = itertools.cycle([""] * 4 + ["Buzz"])
    fizzes_buzzes = (fizz + buzz for fizz, buzz in zip(fizzes, buzzes))
    result = (word or n for word, n in zip(fizzes_buzzes, itertools.count(1)))
    for i in itertools.islice(result, n):
        pass


time_for = timeit.timeit(
    fizz_buzz,
    number=1000)

time_list_comprehension = timeit.timeit(
    fizz_buzz_list_comprehension,
    number=1000)

time_itertools = timeit.timeit(
    fizz_buzz,
    number=1000)


print(
    'For loop: {:.4f}\nList comprehension: {:.4f}\nItertools: {:.4f}'.format(
        time_for, time_list_comprehension, time_itertools
        )
    )


mu = 0.05
sigma = 0.2
T = 1
n_sim = 1000


def MC_call():
    price = 0.0
    for _ in range(n_sim):
        price += np.maximum(mu + sigma * np.sqrt(T) * np.random.randn(), 0)
    return price / n_sim


def MC_call_list_comprehension():
    prices = [
        np.maximum(mu + sigma * np.sqrt(T) * np.random.randn(), 0)
        for _ in range(n_sim)
        ]
    return sum(prices) / n_sim


def MC_call_itertools():
    price = 0.0
    for i in itertools.count(start=0, step=1):
        if i < n_sim:
            price += np.maximum(mu + sigma * np.sqrt(T) * np.random.randn(), 0)
        else:
            break
    return price / n_sim


time_for = timeit.timeit(
    MC_call,
    number=100)

time_list_comprehension = timeit.timeit(
    MC_call_list_comprehension,
    number=100)

time_itertools = timeit.timeit(
    MC_call_itertools,
    number=100)

print(
    'For loop: {:.4f}\nList comprehension: {:.4f}\nItertools: {:.4f}'.format(
        time_for, time_list_comprehension, time_itertools
        )
    )
