class Human:
    def __init__(self):
        self.Sport = 'Running'
        self.Clothes = 'Nike'
        self.Vehicle = 'Audi'
        self.From = 'Hangzhou'

    def where_are_you_from(self):
        return self.From

    def what_are_you_wearing(self):
        return self.Clothes

    def what_is_your_car(self):
        return self.Vehicle

    def what_are_you_doing(self):
        return self.Sport


class Soilder(Human):
    def __init__(self):
        super().__init__()
        self.SpecialFunction = 'Gun'

    def soild_only(self):
        return self.SpecialFunction

class lianmeng:
    def __init__(self):
        self.Q = 'Press Q'
        self.W = 'Press W'
        self.E = 'Press E'
        self.R = 'Press R'
        self.juesexingbie = ''

class NKSS(lianmeng):
    def __init__(self):
        super(NKSS, self).__init__()

    def SkillQ(self):
        return self.Q

class DMXY(lianmeng):
    def __init__(self,a,b,c):
        super(DMXY, self).__init__()
        self.skilla = a
        self.skillb = b
        self.skillc = c
        return 0





Soilder76 = Soilder()
Jonathan = Human()

name = 'Jonathan'
name1 = name.swapcase()
name2 = name.upper()
name3 = name.isupper()


def print_person_imformation():
    print(Jonathan.what_are_you_doing())
    print(Jonathan.what_is_your_car())
    print(Jonathan.what_are_you_wearing())
    print(Jonathan.where_are_you_from())
    print(Soilder76.soild_only())
    print(Soilder76.where_are_you_from())


print_person_imformation()
print(name.swapcase())
print(name1)
print(name2)
print(name3)




