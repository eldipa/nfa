'''
>>> from sm import *
'''

import copy

def move(state, char):
    ''' Return the state that can be reached from the
        state <state> on input <char>, None if such transition
        does not exist.

        >>> s1, s2 = State(), State()
        >>> s1 >> (s2, 'a')

        >>> move(s1, 'a') == s2
        True

        >>> move(s1, 'b') is None
        True

        This is not affected by e-transitions
        >>> s1 >> s2
        >>> move(s1, 'a') == s2
        True
        '''
    return state.next_state_on(char)

def move_s(states, char):
    ''' Return the states that can be reached from any
        state in the <states> set on input <char>.
        Return an empty set if no transition is found.

        >>> s1, s2, s3 = State(), State(), State()
        >>> s1 >> (s2, 'a')
        >>> s2 >> (s3, 'a')
        >>> s1 >> (s3, 'b')

        >>> move_s({s1, s2}, 'a') == {s2, s3}
        True

        >>> move_s({s1, s2}, 'b') == {s3}
        True

        >>> move_s({s1, s2}, 'c') == set()
        True
        '''
    S = frozenset(move(s, char) for s in states)
    return S - {None}

def imm_e_closure(state):
    ''' Set of states reachable from <state> using e-transitions alone.

        >>> s1, s2, s3 = State(), State(), State()
        >>> s1 >> (s2, 'a')
        >>> s1 >> s2
        >>> s2 >> (s1, 'b')
        >>> s2 >> s3

        >>> imm_e_closure(s1) == {s2}
        True

        >>> imm_e_closure(s2) == {s3}
        True

        >>> imm_e_closure(s3) == set()
        True
        '''
    reachable_by_e = frozenset(s for s, x in state.trans if x is None)
    return reachable_by_e

def dbg(set_state, all_states_list):
    s = all_states_list
    return [i for i, si in enumerate(s) if si in set_state]

def e_closure_s(states):
    ''' Return the states that can be reached from some state in <states>
        on e-transitions alone (it is the union of all the imm_e_closure states).

        >>> s1, s2, s3 = State(), State(), State()
        >>> s1 >> (s2, 'a')
        >>> s1 >> s2
        >>> s2 >> (s1, 'b')
        >>> s2 >> s3

        >>> e_closure_s({s1}) == {s1, s2, s3}
        True

        >>> e_closure_s({s2, s3}) == {s2, s3}
        True

        >>> _ = '(a|b)*abb'
        >>> s = [State() for i in range(11)]
        >>> s[0] >> s[1] ; s[0] >> s[7]; s[1] >> s[2]; s[1] >> s[4];
        >>> s[2] >> (s[3], 'a'); s[4] >> (s[5], 'b');
        >>> s[3] >> s[6]; s[5] >> s[6]; s[6] >> s[7]; s[6] >> s[1];
        >>> s[7] >> (s[8], 'a'); s[8] >> (s[9], 'b'); s[9] >> (s[10], 'b')

        >>> e_closure_s({s[0]}) == {s[i] for i in (0, 1, 2, 4, 7)}
        True

        >>> e_closure_s({s[3]}) == {s[3], s[6], s[7], s[1], s[2], s[4]}
        True

        #>>> dbg(e_closure_s({s[3], s[8]}), s)
        >>> e_closure_s({s[3], s[8]}) == {s[i] for i in range(1,9)} - {s[5]}
        True
        '''
    S = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()

        # immediate e-closure of the not-processed-yet state
        tmp = imm_e_closure(s)
        # what new states we discovered?
        new = tmp - S
        # update our stack of unseen states and the final S
        stack.extend(new)
        S |= new

    return frozenset(S)

class State:
    __slots__ = ('trans', 'next_state_by_input', 'next_state_by_func')
    def __init__(self):
        self.trans = set()
        self.next_state_by_input = {}
        self.next_state_by_func = None

    def next_state_on(self, input):
        assert input is not None
        s = self.next_state_by_input.get(input)
        if s is None and self.next_state_by_func is not None:
            next_state, func = self.next_state_by_func
            if func(input):
                s = next_state
        return s

    def __rshift__(self, next):
        ''' Add a transition from self to the state <next>.
            <next> can be a tuple of (<state>, <input>) to create
            a "labeled" transition on <input> to <state>
            If <input> is None or <next> is not a tuple, assume
            an e-transition to the state <next>.

            <input> can be also a function that should return True
            if a particular non-empty input matches and therefor
            this state (self) should transition to <state>, False
            otherwise.
            '''
        try:
            si, input = next
        except:
            si, input = next, None

        assert isinstance(si, State)
        self.trans.add((si, input))

        if input is not None:
            if callable(input):
                assert self.next_state_by_func is None
                self.next_state_by_func = (si, input)
            else:
                self.next_state_by_input[input] = si


class NFA:
    __slots__ = ('i', 'f')

    def __init__(self):
        self.i, self.f = State(), State()

    def __or__(self, other):
        return union(self, other)

    def __ror__(self, other):
        return union(other, self)

    def __getitem__(self, i):
        '''
            Note: the slice notation differs from Python semantics.
            In Python a:b means a range from a (inclusive) to b (exclusive)
            and can be negative and even they can take a step parameter a:b:c
            The slice notation for NFA a:b is inclusive in both ends and cannot
            be negative or take a step parameter.

            Exactly N (repeat)
            >>> sm = concat('a', 'b')
            >>> sm = sm[3]
            >>> [simulate_nfa(sm, st) for st in ('abab', 'ababab', 'abababab')]
            [False, True, False]

            N or more
            >>> sm = concat('a', 'b')
            >>> sm = sm[2:]
            >>> [simulate_nfa(sm, st) for st in ('ab', 'abab', 'ababab')]
            [False, True, True]

            1 or more
            >>> sm = concat('a', 'b')
            >>> sm = sm[1:]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab')]
            [False, True, True]

            0 or more
            >>> sm = concat('a', 'b')
            >>> sm = sm[0:]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab')]
            [True, True, True]

            >>> sm = concat('a', 'b')
            >>> sm = sm[:]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab')]
            [True, True, True]

            0 o 1 (optional)
            >>> sm = concat('a', 'b')
            >>> sm = sm[:1]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab')]
            [True, True, False]

            N o N+1 (repeat+optional)
            >>> sm = concat('a', 'b')
            >>> sm = sm[1:2]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab', 'ababab')]
            [False, True, True, False]

            >>> sm = concat('a', 'b')
            >>> sm = sm[2:3]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab', 'ababab', 'abababab')]
            [False, False, True, True, False]

            N o M (range)
            >>> sm = concat('a', 'b')
            >>> sm = sm[0:2]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab', 'ababab')]
            [True, True, True, False]

            >>> sm = concat('a', 'b')
            >>> sm = sm[1:3]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab', 'ababab', 'abababab')]
            [False, True, True, True, False]

            >>> sm = concat('a', 'b')
            >>> sm = sm[2:4]
            >>> [simulate_nfa(sm, st) for st in ('', 'ab', 'abab', 'ababab', 'abababab', 'ababababab')]
            [False, False, True, True, True, False]
            '''

        if isinstance(i, int):
            return repeat(self, i)
        elif isinstance(i, slice):
            min, max, step = i.start, i.stop, i.step

            # min/max type checking
            if min is not None and not isinstance(min, int):
                raise TypeError("The min must be of type int but it is %s" % type(min))
            if max is not None and not isinstance(max, int):
                raise TypeError("The max must be of type int but it is %s" % type(max))

            # step makes no sense
            if step is not None:
                raise IndexError("The slice must not have a step but it has one %s" % step)

            # min/max int validity
            if min is not None and max is not None and min > max:
                raise IndexError("The max %s is less than the min %s." % (max,min))
            if (min is not None and min < 0):
                raise IndexError("The min must be positive but it is %s" % min)
            if (max is not None and max < 0):
                raise IndexError("The max must be positive but it is %s" % max)

            # min alias
            if min is None:
                min = 0

            # special construction (simpler)
            if max is None and min in (0, 1):
                return klee(self, can_be_zero=min==0)

            prefix = None
            if min >= 1:
                prefix = copy.deepcopy(self)
                if min > 1:
                    prefix = prefix[min]

            postfix = None
            if max is not None:
                postfix = optional(self)
                if max - min > 1:
                    postfix = postfix[max-min]
            else:
                postfix = klee(self, can_be_zero=True)

            if prefix is None:
                return postfix
            else:
                return concat(prefix, postfix)

        else:
            raise IndexError

def L(x):
    ''' Build a state machine with a single transition from its
        initial state to its final state on the given input <x>.

        >>> simulate_nfa(L('a'), "a")
        True

        >>> simulate_nfa(L('a'), "b")
        False

        >>> simulate_nfa(L('a'), "aa")
        False

        >>> simulate_nfa(L('a'), "")
        False
        '''
    sm = NFA()
    sm.i >> (sm.f, x)
    return sm

def as_state_machines(*args):
    sms = [sm if isinstance(sm, NFA) else L(sm) for sm in args]
    return sms[0] if len(sms) == 1 else sms

def union(sm0, sm1):
    ''' Given two state machines <sm0> and <sm1>, build
        a third one as the union of those two.
        In regex terms: <sm0>|<sm1> (or)

        After this, neither <sm0> nor <sm1> can be used in
        other construction.

        >>> simulate_nfa(L('a') | 'b', "a")
        True

        >>> simulate_nfa(L('a') | 'b', "b")
        True

        >>> simulate_nfa(L('a') | 'b', "ab")
        False

        >>> simulate_nfa(L('a') | 'b', "")
        False
        '''
    sm0, sm1 = as_state_machines(sm0, sm1)
    assert sm0 != sm1
    sm = NFA()

    # from the initial state go to sm0 *or* sm1
    sm.i >> sm0.i
    sm.i >> sm1.i

    # then, from sm0 *or* sm1 go to final state
    sm0.f >> sm.f
    sm1.f >> sm.f

    return sm

def concat(*args):
    ''' Build a state machine as the concatenation of each
        state machine in <args> (left to right)
        In regex terms: <sm0><sm1><sm2>... (concatenation)

        After this, none of the state machine <smi> can be used in
        other construction.

        >>> simulate_nfa(concat(L('a'), 'b'), "ab")
        True

        >>> simulate_nfa(concat(L('a'), 'b'), "abx")
        False

        >>> simulate_nfa(concat(L('a'), 'b'), "xab")
        False
        '''
    sm = NFA()
    args = as_state_machines(*args)

    # go from si to sm0
    sm.i >> args[0].i
    prev_sm = args[0]

    # then from sm(i-1) to sm(i)
    for smi in args[1:]:
        prev_sm.f >> smi.i
        prev_sm = smi

    # finally, the last sm(i) go to the final state
    prev_sm.f >> sm.f

    return sm

def repeat(sm0, ntimes):
    ''' Build a state machine as the concatenation of <sm0> with itself
        <ntimes>.
        In regex terms: <sm0>{n} (repeat)

        Note that <ntimes> must be greater than or equal to 1.

        After this, <sm0> cannot be used in other construction.

        >>> simulate_nfa(L('a')[3], "aaa")
        True

        >>> simulate_nfa(L('a')[3], "aaaa")
        False

        >>> simulate_nfa(L('a')[3], "aa")
        False
        '''
    sm0 = as_state_machines(sm0)
    assert ntimes >= 1
    sm = NFA()

    sm_prev = sm0
    sm.i >> sm_prev.i

    for _ in range(ntimes-1):
        sm_next = copy.deepcopy(sm0)
        sm_prev.f >> sm_next.i
        sm_prev = sm_next

    sm_prev.f >> sm.f

    return sm


def optional(sm0):
    ''' Build a state machine that matches <sm0> or not (<sm0> is optional).
        In regex terms: <sm0>? (optional)

        After this, <sm0> cannot be used in other construction.

        >>> simulate_nfa(L('a')[:1], "a")
        True

        >>> simulate_nfa(L('a')[:1], "")
        True
        '''
    sm0 = as_state_machines(sm0)
    sm0.i >> sm0.f
    return sm0

def klee(sm0, can_be_zero):
    ''' Build a state machine that matches <sm0> zero or more
        (if <can_be_zero> is True) or one or more (if <can_be_zero> is False).
        In regex terms: <sm0>* (zero or more) or <sm0>+ (one or more)

        After this, <sm0> cannot be used in other construction.

        NFA's alias of klee with can_be_zero==True
        >>> simulate_nfa(L('a')[:], "a")
        True

        >>> simulate_nfa(L('a')[:], "b")
        False

        >>> simulate_nfa(L('a')[:], "aa")
        True

        >>> simulate_nfa(L('a')[:], "")
        True

        NFA's alias of klee with can_be_zero==False
        >>> simulate_nfa(L('a')[1:], "a")
        True

        >>> simulate_nfa(L('a')[1:], "b")
        False

        >>> simulate_nfa(L('a')[1:], "aa")
        True

        >>> simulate_nfa(L('a')[1:], "")
        False
        '''
    sm0 = as_state_machines(sm0)

    # loop to itself
    sm0.f >> sm0.i

    if can_be_zero:
        optional(sm0)

    return sm0

def simulate_nfa(sm, string):
    '''
        Simulate or run the nondeterministic finite automata <sm>
        feeding it with the characters of the <string>.

        >>> sm = concat('a', 'b', 'b')
        >>> simulate_nfa(sm, "abb")
        True

        >>> sm = concat(L('a')[:], 'a', 'b', 'b')
        >>> simulate_nfa(sm, "abb")
        True

        >>> simulate_nfa(sm, "aabb")
        True

        # '(a|b)*abb'
        >>> sm = concat((L('a') | 'b')[:], 'a', 'b', 'b')

        >>> simulate_nfa(sm, "aabbabb")
        True

        >>> simulate_nfa(sm, "aaabb")
        True

        >>> simulate_nfa(sm, "aaabba")
        False

        The NFA simulation supports other alphabets beside the
        traditional ASCII.

        In particular, a valid item can be:
            - a multi bytes item (string)
            - a number
            - a frozen set

        Technically any hasheable object will work but their semantics
        are reserved.

        >>> obj1 = frozenset()
        >>> obj2 = frozenset({1})
        >>> sm = concat('Symbol', '=', 1, obj1)
        >>> simulate_nfa(sm, ['Symbol', '=', 1, obj1])
        True

        Functions wrapped in a L construction can be used to define
        a family o class of items, like "any digit".
        >>> sm = concat(L(str.isdigit)[1:], str.isalpha)
        >>> simulate_nfa(sm, "1a")
        True
        >>> simulate_nfa(sm, "ba")
        False
        >>> simulate_nfa(sm, "123a")
        True
        '''
    S = e_closure_s({sm.i})

    eof = False
    for c in string:
        if not S:
            break
        M = move_s(S, c)
        S = e_closure_s(M)
    else:
        eof = True

    return eof and sm.f in S

