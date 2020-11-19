from __future__ import division, print_function

from collections import Counter

import sympy
from sympy import Add, Expr, Function, Mul, Pow, Rational, S, factorial, symbols
from sympy.core.decorators import _sympifyit, call_highest_priority
from sympy.core.function import UndefinedFunction
from sympy.printing.latex import print_latex

from Poisson import PoissonBracket

# TODO: double dot notation not correct when expanding the poisson bracket has to go in the bracket


class LieOperator(Expr):
    """
    Class defining Lie operators.
    """

    _op_priority = 11.0
    is_commutative = False

    def __new__(cls, ham, indep_coords, indep_mom):
        # create list of coords
        if not isinstance(indep_coords, list):
            indep_coords = [indep_coords]

        # create list of momenta
        if not isinstance(indep_mom, list):
            indep_mom = [indep_mom]

        # same number of coordinates and momenta have to be entered
        assert len(indep_coords) == len(
            indep_mom
        ), "The number of coords and momenta is not the same"

        obj = Expr.__new__(cls, ham, indep_coords, indep_mom)
        obj._ham = ham
        obj._indep_coords = indep_coords
        obj._indep_mom = indep_mom

        return obj

    @property
    def ham(self):
        return self._ham

    @ham.setter
    def ham(self, other):
        # Input has to be a function
        assert not (S(other.is_number)), "Input has to be a function and not a number"
        self._ham = self._ham.subs(self.ham, other).doit()

    @property
    def indep_coords(self):
        return self._indep_coords

    @indep_coords.setter
    def indep_coords(self, other):
        self._indep_coords = other

    @property
    def indep_mom(self):
        return self._indep_mom

    @indep_mom.setter
    def indep_mom(self, other):
        self._indep_mom = other

    # Define Addition
    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__radd__")
    def __add__(self, other):
        if not isinstance(other, LieOperator):
            raise TypeError(other, " has to be a Lie Operator")
        elif not Counter(self.indep_coords) == Counter(other.indep_coords):
            raise TypeError("Lie Operators do not have the same dimension.")
        else:
            return LieOperator(self.ham + other.ham, self.indep_coords, self.indep_mom)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__add__")
    def __radd__(self, other):
        if not isinstance(other, LieOperator):
            raise TypeError(other, " has to be a Lie Operator")
        elif not Counter(self.indep_coords) == Counter(other.indep_coords):
            raise TypeError("Lie Operators do not have the same dimension.")
        else:
            return LieOperator(self.ham + other.ham, self.indep_coords, self.indep_mom)

    # Define Substraction
    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__rsub__")
    def __sub__(self, other):
        if not isinstance(other, LieOperator):
            raise TypeError(other, " has to be a Lie Operator")
        elif not Counter(self.indep_coords) == Counter(other.indep_coords):
            raise TypeError("Lie Operators do not have the same dimension.")
        else:
            return LieOperator(self.ham - other.ham, self.indep_coords, self.indep_mom)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__sub__")
    def __rsub__(self, other):
        if not isinstance(other, LieOperator):
            raise TypeError(other, " has to be a Lie Operator")
        elif not Counter(self.indep_coords) == Counter(other.indep_coords):
            raise TypeError("Lie Operators do not have the same dimension.")
        else:
            return LieOperator(self.ham - other.ham, self.indep_coords, self.indep_mom)

    # Define Multiplication
    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__rmul__")
    def __mul__(self, other):
        return self.LieOperatorMul(self, other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__mul__")
    def __rmul__(self, other):
        if S(other).is_number:
            return LieOperator(other * self.ham, self.indep_coords, self.indep_mom)
        elif not isinstance(other, LieOperator):
            other = LieOperator(other, self.indep_coords, self.indep_mom)
        return other.LieOperatorMul(self)

    # @staticmethod
    def LieOperatorMul(self, _ham1, _ham2):
        if not isinstance(_ham2, LieOperator):
            _ham2 = LieOperator(_ham2, _ham1.indep_coords, _ham1.indep_mom)
        if not Counter(_ham1.indep_coords) == Counter(_ham2.indep_coords):
            raise TypeError("Lie Operators do not have the same dimension.")
        else:
            return LieOperator(
                PoissonBracket(
                    _ham1.ham,
                    _ham2.ham,
                    coords=_ham1.indep_coords,
                    mom=_ham1.indep_mom,
                ),
                _ham1.indep_coords,
                _ham1.indep_mom,
            )

    # Calcualte Poisson bracket with input function
#     def Poisson(self, other):
#         _h1 = self.ham
#         _h2 = other.ham
#         hp = S(0)
#         for qi, pi in zip(self.indep_coords, self.indep_mom):
#             hp += _h1.diff(qi) * _h2.diff(pi)
#             hp -= _h1.diff(pi) * _h2.diff(qi)
#         return LieOperator(hp, self.indep_coords, self.indep_mom)

    # Exponential Map: Lie Transform Map
    # Calculate the exponential map of the Lie operator to the input cutoff
    def LieMap(self, other, power):
        s = S(0)

        for i in range(power + 1):
            s += Rational(1, factorial(i)) * (self.ExpPowerLieBracket(other, i)).ham

        return LieOperator(s, self.indep_coords, self.indep_mom)

    # Successively apply Poisson bracket to input function to the input cutoff
    def ExpPowerLieBracket(self, other, power):
        _op1 = self
        _op2 = other

        if power > 0:
            hp = _op1 * _op2

            for s in range(1, power):
                hp = _op1 * hp

        else:
            if not isinstance(_op2, LieOperator):
                _op2 = LieOperator(_op2, self.indep_coords, self.indep_mom)

            hp = _op2

        return hp

    # Up to order 4 done manually to have a working copy, arbitrary ordered needs to be still written
    def BCH(self, other, n):
        if not isinstance(other, LieOperator):
            raise TypeError(other, " has to be a Lie Operator")
        elif not Counter(self.indep_coords) == Counter(other.indep_coords):
            raise TypeError("Lie Operators do not have the same dimension.")
        elif n < 1:
            raise ValueError(n, " is not a valid order. Number has to be natural.")

        _op1 = LieOperator(self.ham, self.indep_coords, self.indep_mom)
        _op2 = LieOperator(other.ham, other.indep_coords, other.indep_mom)

        temp = _op1 + _op2

        if n > 1:
            _xy = _op1 * _op2
            _yx = _op2 * _op1
            temp = temp + Rational(1, 2) * _xy

        if n > 2:
            _xxy = _op1 * _xy
            _yyx = _op2 * _yx
            temp = temp + Rational(1, 12) * _xxy + Rational(1, 12) * _yyx

        if n > 3:
            _yxxy = _op2 * _xxy
            temp = temp - Rational(1, 24) * _yxxy

        if n > 4:
            _xyxyx = _op1 * (_op2 * (_op1 * _yx))
            _yxyxy = _op2 * (_op1 * (_op2 * _xy))

            _xxxy = _op1 * _xxy
            _yyyx = _op2 * _yyx
            temp = temp + (
                Rational(1, 120) * _xyxyx
                + Rational(1, 120) * _yxyxy
                + Rational(1, 360) * _op1 * _yyyx
                + Rational(1, 360) * _op2 * _xxxy
                - Rational(1, 720) * _op2 * _yyyx
                - Rational(1, 720) * _op1 * _xxxy
            )

        return temp

    def _latex(self, printer, *args):
        # print(self._ham.__class__.__name__)
        # print(len(self.ham.args))
        # print(self.ham.args)
        # news = []
        # for arg in [self.ham]:
        #    if isinstance(arg, Add):
        #        news.append(
        #            " + ".join(
        #                [
        #                    printer._print(a.func, *args)
        #                    if a.is_Function
        #                    else printer._print(a, *args)
        #                    for a in arg.args
        #                ]
        #            )
        #        )

        #    elif isinstance(arg, PoissonBracket):
        #        news.append(printer.doprint(arg, *args))

        #    elif isinstance(arg, Mul):
        #        news.append(" * ".join([printer._print(arg, *args) for arg in self.args[:2]]))

        #    elif isinstance(arg, Pow):
        #        if isinstance(arg.args[0], UndefinedFunction):
        #            news.append("{}^{}".format([printer._print(a) for a in arg.args]))
        #        elif isinstance(arg.args[0], Function):
        #            news.append(
        #                "{}^{}".format(
        #                    printer._print(arg.args[0].func, *args),
        #                    printer._print(arg.args[1], *args),
        #                )
        #            )
        #        else:
        #            news.append(printer._print(arg, *args))

        #    elif isinstance(arg, Function) and not (isinstance(arg, UndefinedFunction)):
        #        news.append(printer._print(arg.func, *args))

        #    else:  # isinstance(arg, Function) and isinstance(arg,UndefinedFunction):
        #        news.append(printer._print(arg, *args))

        # return "%s" % tuple(news)

        ##### prints ham with functions as ham has functions with own print method, need to overwrite
        if self._ham.__class__.__name__ == "Add":
            if len(self.ham.args) >= 2:
                pham = (
                    printer.doprint(self._ham.args[-2].func)
                    + "+"
                    + printer.doprint(self._ham.args[-1].func)
                )
                if len(self.ham.args) > 2:
                    pham += " + "
                    pham += printer.doprint(
                        (self.ham - self.ham.args[-2] - self.ham.args[-1]).as_expr()
                    )
            else:
                pham = printer.doprint(self.ham.as_expr())
        elif isinstance(self.ham, UndefinedFunction):
            pham = printer.doprint(self._ham)
        else:
            pham = printer.doprint(self._ham.func)

        return r"e^{{:{}:}}".format(pham)
