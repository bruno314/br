from random import randint, shuffle
from collections import defaultdict, namedtuple, Counter, deque
from operator import itemgetter
from pprint import pprint
from queue import Queue, PriorityQueue
from pandas import DataFrame, Series

import unicodedata, sys, re, time, gc, os, math, pickle, webbrowser, requests, tempfile, random, \
    itertools as it, \
    numpy as np, \
    pandas as pd

___TAG_RE = re.compile(r'<[^>]+>')

def generic_test_engine(input_matrix, callback, limit=sys.maxsize, postprocess=None, unpack=False):
    """
    : unpack=True interpretuje prvy argument ako pole jednotlivych argumentov predatelnych funkcii [a,b,c] -> f(a,b,c)
    """
    statuses = []
    no_all = 0
    no_correct = 0

    for i, single_test_case in enumerate(input_matrix):
        if i == limit: break

        test_case_in = single_test_case[0]
        test_case_correct_out = single_test_case[1]

        if not unpack:
            tested_function_output = callback(test_case_in)
        if unpack:
            tested_function_output = callback(*test_case_in)

        no_all += 1
        if postprocess is not None:
            tested_function_output_postprocessed = postprocess(tested_function_output)
        else:
            tested_function_output_postprocessed = tested_function_output

        # co ak sme zabudli pretypovat? Toto nas upozorni!
        if type(tested_function_output_postprocessed) != type(test_case_correct_out):
            statuses.append('[WARNING] types of compared variables are NOT same')

        if tested_function_output_postprocessed == test_case_correct_out:
            statuses.append("{} OK".format(i))
            no_correct += 1

        else:
            statuses.append("{index} FAILED In: {test_case_in} \n Our: {our} \n Correct: {correct}".format(
                index=i,
                test_case_in=test_case_in,
                our=tested_function_output,
                correct=test_case_correct_out
            )
            )

    print("""
         ----------------------------------------------------
         RESULT {no_correct}/{no_all} passed.
         ----------------------------------------------------
    """.format(
        no_correct=no_correct,
        no_all=no_all

    ))

    print("\n".join(statuses))


def isint(param):
    # returns true if param is int
    return type(param) == type(100)
def isstr(param):
    return type(param) == type(" ")
def islist(param):
    return type(param) == type([1, 2, 3])
def strip_tags(text):
    return ___TAG_RE.sub('', text)
def arr_2d(rows, cols, default_value=None):
    return [x[:] for x in [[default_value] * cols] * rows]
def ___gen_from_lol(buffer, header=None):
    header_line = "<td>" + "</td><td>".join(map(str, header)) + "</td>" if header is not None else \
        "<td>HDR</td>" * (len(buffer[0]) + 2)

    return "<table>" + header_line \
           + "".join(
        [
            "<tr><td>" + "</td><td>".join(map(str, line)) + "</td></tr>" for line in buffer
        ]) \
           + '</table>'
def pr(*args, **kwargs):
    for index, argument in enumerate(args):
        print(index, pprint(argument))  # !! (())

    for key, argument in kwargs.items():
        print("{}=\n{}".format(key, argument))
def cutedebug(what, size=9):
    if type(what) == type([]):
        # it's a list!
        tempfile_name = tempfile.mktemp(suffix='.html')
        pr(tempfile_name)

        first_type = type(what[0])
        is_homog = True

        for index, item in enumerate(what):
            if index == 0: continue

            if type(item) != first_type:
                is_homog = False

        if is_homog:
            # print ("first_type = {} ".format(first_type))


            if first_type == type([]):
                # pole x pole
                header = None
                buffer = what

            if first_type == type({}):
                # pole dictov
                header = list(what[0].keys())
                buffer = [g.values() for g in what]

            with open(tempfile_name, "w") as tempfile_handle:
                tempfile_handle.write(___gen_from_lol(buffer, header=header))

            webbrowser.open(tempfile_name, new=2)
            time.sleep(1)
            os.remove(tempfile_name)


            # vykresli
        else:
            Exception("NON HOMOGENOUS (cutedebug) ")
def arr(rows):
    return [None] * rows
def input_integers():  # for any number of space separated integers
    return_list = list(map(int, input().split(" ")))
    if len(return_list) == 1:
        return return_list[0]
    else:
        return return_list
def var_exists(variable_name):
    return (variable_name in vars() or variable_name in globals())
def remove_punctuation(string):
    return str(unicodedata.normalize('NFKD', str(string)).encode('ASCII', 'ignore').decode("utf-8"))
class postfix:
    class sqrt():
        def __rrshift__(self, other):
            return math.sqrt(other)

    class len:
        def __init__(self):
            pass

        def __rrshift__(self, other):
            return len(other)

    class type:
        def __init__(self):
            pass

        def __rrshift__(self, other):
            return type(other)

    class list:
        def __init(self):
            pass

        def __rrshift__(self, other):
            return list(other)

    class printer:
        def __init__(self):
            self.show = True

        def __rrshift__(self, other):  # single argument - postfix unary only
            if self.show:
                print(other)
            return other  # SOOO IMPORTANT!

# quick postfix printer creation

class p:
    pr = postfix.printer()
    len = postfix.len()
    type = postfix.type()
    list = postfix.list()

    def __new__(cls, *args, **kwargs):
        print(args[0])
        return False

def brake():
    """terminates"""
    print("""
    -----------------------
    | OMG STOP             |
    -----------------------
    """)
    sys.exit(0)
def var_save(**kwargs):
    if len(kwargs.items()) != 1:
        print("UR A FAILURE HAHAHA")
    else:
        for filename, variable in kwargs.items():
            with open("{}.pickle".format(filename), "wb") as _filehandle:
                pickle.dump(variable, _filehandle)
def var_load(filename):
    with open("{}.pickle".format(filename), "rb") as ___filehandle:
        return pickle.load(___filehandle)

class ProgressTracker():
    def __init__(self, max_steps, number_of_messages=20, clear_console=False):
        self.max_steps = max_steps
        self.current_step = 0
        self.number_of_messages = number_of_messages
        self.clear_console = clear_console

    def display_message(self):
        if self.clear_console:
            print('\n' * 100)

        print("[{}%] {}/{}".format(100 * self.current_step // self.show_after_how_much / self.number_of_messages,
                                   self.current_step // self.show_after_how_much, self.number_of_messages))

    def one_step(self):
        self.current_step += 1
        self.show_after_how_much = self.max_steps // self.number_of_messages
        if self.current_step % self.show_after_how_much == 0:
            self.display_message()

def dimension_test(*args):
    """
    Prints dimensions of (nested) iterable structure.
    Equivalent to:
        for  variable in arguments print
            len(variable)
            len(variable[0])
            len(variable[0][0])
            ...
            until variable[0]...[0] has meaningful length at least 1

    :param args:
    :return:
    """
    for c, var in enumerate(args):
        if len(var) < 1: continue
        print("#{}".format(c))
        lvl = 1
        has_next_level = True
        currentReference = var

        while has_next_level:
            print("--- " * lvl, ">", len(currentReference))

            try:
                next_level_len = len(currentReference[0])
            except TypeError:
                next_level_len = 0

            if next_level_len > 1:
                has_next_level = True
                currentReference = currentReference[0]
            else:
                has_next_level = False

            lvl += 1

def dimension_test_test():
    d0 = []
    d1 = [1, 2, 3]
    d2 = [[11, 12], [21, 22], [31, 32]]
    d3 = [[[i * j * k for i in range(2)] for j in range(3)] for k in range(4)]
    dimension_test(d0, d1, d2, d3)
