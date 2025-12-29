
class OffsetAlgebraError(ValueError): pass

class DimensionError(ValueError):
    def __init__(self, u_from, u_to, dim_from, dim_to):
        msg = ("Incompatible dimensions: '%s' %s → '%s' %s. "
               "Units must have identical dimension vectors.") % (
                   u_from, dim_from, u_to, dim_to
               )
        ValueError.__init__(self, msg)
        self.u_from = u_from
        self.u_to = u_to
        self.dim_from = dim_from
        self.dim_to = dim_to

class UnknownUnitError(ValueError):
    def __init__(self, token):
        ValueError.__init__(self, "Unknown unit symbol: '%s'" % token)
        self.token = token


def _assert_linear(u):
    if getattr(u, 'offset', 0) not in (0, 0.0):
        raise OffsetAlgebraError(f"Algebra with offset unit '{u.symbol}' is not allowed."
                                 "Normaize temperature to k or degR before multiplying/dividing/raising to a power")
              


class PrefixDefinition:
    def __init__(self, symbol, name, factor):
        self.symbol = symbol
        self.name = name
        self.factor = factor

    def __repr__(self):
        return f"Prefix(symbol='{self.symbol}', name='{self.name}', factor={self.factor})"

    def __eq__(self, other):
        return isinstance(other, PrefixDefinition) and self.factor == other.factor

    def __mul__(self, unit):
        if isinstance(unit, UnitDefinition):
            return UnitDefinition(symbol=self.symbol + unit.symbol, name=self.name + unit.name, 
                                  L=unit.L, M=unit.M, T=unit.T, I=unit.I, THETA=unit.THETA, N=unit.N, J=unit.J, 
                                  coef=self.factor * unit.coef, 
                                  offset=unit.offset)
        else:
            raise ValueError("Can only multiply with UnitDefinition.")

PREFIXES = {
    'y':  PrefixDefinition(symbol='y',  name='yocto', factor=1E-24),
    'z':  PrefixDefinition(symbol='z',  name='zepto', factor=1E-21),
    'a':  PrefixDefinition(symbol='a',  name='atto',  factor=1E-18),
    'f':  PrefixDefinition(symbol='f',  name='femto', factor=1E-15),
    'p':  PrefixDefinition(symbol='p',  name='pico',  factor=1E-12),
    'n':  PrefixDefinition(symbol='n',  name='nano',  factor=1E-09),
    'µ':  PrefixDefinition(symbol='µ',  name='micro', factor=1E-06),
    'u':  PrefixDefinition(symbol='u',  name='micro', factor=1E-06),
    'm':  PrefixDefinition(symbol='m',  name='milli', factor=1E-03),
    'c':  PrefixDefinition(symbol='c',  name='centi', factor=1E-02),
    'd':  PrefixDefinition(symbol='d',  name='deci',  factor=1E-01),
    '':   PrefixDefinition(symbol='',   name='',      factor=1E0),
    'da': PrefixDefinition(symbol='da', name='deca',  factor=1E+01),
    'h':  PrefixDefinition(symbol='h',  name='hecto', factor=1E+02),
    'k':  PrefixDefinition(symbol='k',  name='kilo',  factor=1E+03),
    'M':  PrefixDefinition(symbol='M',  name='mega',  factor=1E+06),
    'G':  PrefixDefinition(symbol='G',  name='giga',  factor=1E+09),
    'T':  PrefixDefinition(symbol='T',  name='tera',  factor=1E+12),
    'P':  PrefixDefinition(symbol='P',  name='peta',  factor=1E+15),
    'E':  PrefixDefinition(symbol='E',  name='exa',   factor=1E+18),
    'Z':  PrefixDefinition(symbol='Z',  name='zetta', factor=1E+21),
    'Y':  PrefixDefinition(symbol='Y',  name='yotta', factor=1E+24)
}

class UnitDefinition:
    def __init__(self, symbol, name, L=0, M=0, T=0, I=0, THETA=0, N=0, J=0, coef=1, offset=0):
        self.symbol = symbol
        self.name = name
        self.L = L
        self.M = M
        self.T = T
        self.I = I
        self.THETA = THETA
        self.N = N
        self.J = J
        self.coef = coef
        self.offset = offset

    def __repr__(self):
        return f"Unit(symbol='{self.symbol}', name='{self.name}', coef={self.coef}, offset={self.offset})"

    def __eq__(self, other):
        return (
            isinstance(other, UnitDefinition) and
            self.L == other.L and
            self.M == other.M and
            self.T == other.T and
            self.I == other.I and
            self.THETA == other.THETA and
            self.N == other.N and
            self.J == other.J and
            self.coef == other.coef and
            self.offset == other.offset
        )

    def __mul__(self, other):
        if isinstance(other, UnitDefinition):
            _assert_linear(self)
            _assert_linear(other)
            return UnitDefinition(
                symbol=f"{self.symbol}*{other.symbol}",
                name=f"{self.name}*{other.name}",
                L=self.L + other.L,
                M=self.M + other.M,
                T=self.T + other.T,
                I=self.I + other.I,
                THETA=self.THETA + other.THETA,
                N=self.N + other.N,
                J=self.J + other.J,
                coef=self.coef * other.coef,
                offset=0
            )
        raise ValueError("Can only multiply with UnitDefinition by UnitDefinition.")

    def __pow__(self, power):
        try:
            power = float(power)
        except ValueError:
            raise ValueError("Power must be a number.")
        return UnitDefinition(
            symbol=f"{self.symbol}^{power}",
            name=f"{self.name}^{power}",
            L=self.L * power,
            M=self.M * power,
            T=self.T * power,
            I=self.I * power,
            THETA=self.THETA * power,
            N=self.N * power,
            J=self.J * power,
            coef=self.coef ** power,
            offset=self.offset * power
        )

    def __truediv__(self, other):
        if isinstance(other, UnitDefinition):
            _assert_linear(self)
            _assert_linear(other)
            return UnitDefinition(
                symbol=f"{self.symbol}/{other.symbol}",
                name=f"{self.name}/{other.name}",
                L=self.L - other.L,
                M=self.M - other.M,
                T=self.T - other.T,
                I=self.I - other.I,
                THETA=self.THETA - other.THETA,
                N=self.N - other.N,
                J=self.J - other.J,
                coef=self.coef / other.coef,
                offset=0
            )
        else:
            raise ValueError("Can only divide with UnitDefinition.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return UnitDefinition(
                symbol=f"{other}/{self.symbol}",
                name=f"{other}/{self.name}",
                L=-self.L,
                M=-self.M,
                T=-self.T,
                I=-self.I,
                THETA=-self.THETA,
                N=-self.N,
                J=-self.J,
                coef=other / self.coef,
                offset=other - self.offset
            )
        else:
            raise ValueError("Can only divide with numeric value.")

    def is_same_dimension(self, other):
        return (
            self.L == other.L and
            self.M == other.M and
            self.T == other.T and
            self.I == other.I and
            self.THETA == other.THETA and
            self.N == other.N and
            self.J == other.J
        )

    def convert_to(self, other, value):
        if not self.is_same_dimension(other):
            raise ValueError("Units are not compatible for conversion.")
        base_value = (value + self.offset) * self.coef
        converted_value = base_value / other.coef - other.offset
        return converted_value

UNITS = {
    # Base SI units
    'm':   UnitDefinition('m',   'meter',   L=1),
    'g':   UnitDefinition('g',   'gram',    M=1, coef=1E-3),
    's':   UnitDefinition('s',   'second',  T=1),
    'A':   UnitDefinition('A',   'ampere',  I=1),
    'K':   UnitDefinition('K',   'kelvin',  THETA=1),
    'mol': UnitDefinition('mol', 'mole',    N=1),
    'cd':  UnitDefinition('cd',  'candela', J=1),

    # Derived SI units
    'Hz':  UnitDefinition('Hz',  'hertz', T=-1),
    'N':   UnitDefinition('N',   'newton', M=1, L=1, T=-2),
    'Pa':  UnitDefinition('Pa',  'pascal', M=1, L=-1, T=-2),
    'J':   UnitDefinition('J',   'joule', M=1, L=2, T=-2),
    'W':   UnitDefinition('W',   'watt', M=1, L=2, T=-3),
    'C':   UnitDefinition('C',   'coulomb', T=1, I=1),
    'V':   UnitDefinition('V',   'volt', M=1, L=2, T=-3, I=-1),
    'Ω':   UnitDefinition('Ω',   'ohm', M=1, L=2, T=-3, I=-2),
    'S':   UnitDefinition('S',   'siemens', M=-1, L=-2, T=3, I=2),
    'F':   UnitDefinition('F',   'farad', M=-1, L=-2, T=4, I=2),
    'T':   UnitDefinition('T',   'tesla', M=1, T=-2, I=-1),
    'Wb':  UnitDefinition('Wb',  'weber', M=1, L=2, T=-2, I=-1),
    'H':   UnitDefinition('H',   'henry', M=1, L=2, T=-2, I=-2),
    '°C':  UnitDefinition('°C',  'celsius', THETA=1, offset=273.15),
    'degC':  UnitDefinition('°C',  'celsius', THETA=1, offset=273.15),
    'rad': UnitDefinition('rad', 'radian'),
    'sr':  UnitDefinition('sr',  'steradian'),
    'lm':  UnitDefinition('lm',  'lumen', J=1),
    'lx':  UnitDefinition('lx',  'lux', L=-2, J=1),
    'Bq':  UnitDefinition('Bq',  'becquerel', T=-1),
    'Gy':  UnitDefinition('Gy',  'gray', L=2, T=-2),
    'Sv':  UnitDefinition('Sv',  'sievert', L=2, T=-2),
    'kat': UnitDefinition('kat', 'katal', T=-1, N=1),
    'L':   UnitDefinition('L', 'liter', L=3, coef=1e-3),
    'P':   UnitDefinition('P', 'poise', M=1, L=-1, T=-1, coef=1e-1),

    # Customary US Unit System
    '°F': UnitDefinition('°F', 'fahrenheit', THETA=1, offset=459.67, coef=5/9),
    '°R': UnitDefinition('°R', 'rankin', THETA=1, offset=0, coef=5/9),
    'degF': UnitDefinition('°F', 'fahrenheit', THETA=1, offset=459.67, coef=5/9),
    'degR': UnitDefinition('°R', 'rankin', THETA=1, offset=0, coef=5/9),
    'gal': UnitDefinition('gal', 'gallon', L=3, offset=0, coef=0.00378541),

    'psi':      UnitDefinition('psi', 'psi', M=1, L=-1, T=-2, coef=6894.76),
    'lb':       UnitDefinition('lb', 'pound', M=1, coef=0.453592),
    'lbf':      UnitDefinition('lbf', 'lbf', M=1, L=1, T=-2, coef=4.44822),
    'thou':     UnitDefinition('th', 'thou', L=1, coef=2.54E-5),
    'inch':     UnitDefinition('in', 'inch', L=1, coef=2.54E-2),
    'in':       UnitDefinition('in', 'inch', L=1, coef=2.54E-2),
    'ft':       UnitDefinition('ft', 'foot', L=1, coef=3.048E-1),
    'yard':     UnitDefinition('yd', 'yard', L=1, coef=9.144E-1),
    'chain':    UnitDefinition('ch', 'chain', L=1, coef=20.1168),
    'furlong':  UnitDefinition('fur', 'furlong', L=1, coef=201.168),
    'mile':     UnitDefinition('ml', 'mile', L=1, coef=1609.344),
    'league':   UnitDefinition('lea', 'league', L=1, coef=4828.032),
    'BTU':      UnitDefinition('BTU', 'btu', M=1, L=2, T=-2, coef=1055.06),

    # Miscellaneous units
    'Torr': UnitDefinition('Torr', 'Torr', M=1, L=-1, T=-2, coef=133.322),
    'psia':     UnitDefinition('psi', 'psi', M=1, L=-1, T=-2, coef=6894.76),
    'bar': UnitDefinition('bar', 'bar', M=1, L=-1, T=-2, coef=1E5),
    'min': UnitDefinition('min', 'minute', T=1, coef=60),
    'h': UnitDefinition('h', 'hour', T=1, coef=3600),
}

class UnitParser:
    def __init__(self, unit_str):
        self.unit_str = unit_str

    def parse(self):
        parts = [p.strip() for p in self.unit_str.split('/')]
        numerator, denominators = parts[0], parts[1:]
        parsed_units = self._parse_units(numerator)
        for denominator in denominators:
            for unit in self._parse_units(denominator):
                parsed_units.append(unit ** -1)
        result = parsed_units[0]
        for u in parsed_units[1:]:
            result = result * u
        return result

    def _parse_units(self, units_str):
        units = [u.strip() for u in units_str.split('*') if u.strip()]
        return [self._parse_unit(u) for u in units]


    def _parse_unit(self, unit):
        unit = unit.strip()
        if '^' in unit:
            unit, power = unit.split('^')
            power = float(power)
        else:
            power = 1.0
        return self._parse_simple_unit(unit) ** power

    def _parse_simple_unit(self, unit_s):
        unit_s = unit_s.strip()
    
        # 1) Exact match first (avoids 'min' -> 'm' + 'in', etc.)
        if unit_s in UNITS:
            return UNITS[unit_s]
    
        # 2) Try prefix + base-unit decomposition
        #    Sort prefixes by length (desc) so 'da' beats 'd'
        for prefix in sorted(PREFIXES.keys(), key=len, reverse=True):
            if unit_s.startswith(prefix):
                base_unit = unit_s[len(prefix):]
                if base_unit in UNITS:
                    unit = UNITS[base_unit]
                    pref = PREFIXES[prefix]
                    return pref * unit
    
        raise ValueError(f"Unit '{unit_s}' not recognized")


def convert(value, from_unit_str, to_unit_str):
    if not isinstance(value, (int, float)):
        raise TypeError("Value must be an int or float.")
    
    parser_from = UnitParser(from_unit_str)
    parser_to = UnitParser(to_unit_str)
    
    from_unit = parser_from.parse()
    to_unit = parser_to.parse()
    
    def _dims(u):
        return (u.L, u.M, u.T, u.I, u.THETA, u.N, u.J)
    
    #print("FROM:", from_unit_str, "dims:", _dims(from_unit))
    #print("TO  :", to_unit_str,   "dims:", _dims(to_unit))

    return from_unit.convert_to(to_unit, value)

if __name__ == '__main__':
    # Example conversions
        value = 14.7
        from_unit = '°C' #'°C'
        to_unit = '°R'
        result = convert(value, from_unit, to_unit)
        print(f'{value} {from_unit} is {result} {to_unit}')

        from_unit = '°R' #'°C'
        to_unit = '°C'
        result = convert(value, from_unit, to_unit)
        print(f'{value} {from_unit} is {result} {to_unit}')
        
        from_unit = 'Torr' 
        to_unit = 'Pa'
        result = convert(value, from_unit, to_unit)
        print(f'{value} {from_unit} is {result} {to_unit}')
        
        from_unit = 'psi' 
        to_unit = 'Torr'
        result = convert(value, from_unit, to_unit)
        print(f'{value} {from_unit} is {result} {to_unit}')
        
        value = 1
        from_unit = 'm^3' 
        to_unit = 'gal'
        result = convert(value, from_unit, to_unit)
        print(f'{value} {from_unit} is {result} {to_unit}')
        
        value = 1
        from_unit = 'm^3/s' 
        to_unit = 'gal/min'
        result = convert(value, from_unit, to_unit)
        print(f'{value} {from_unit} is {result} {to_unit}')
        

        