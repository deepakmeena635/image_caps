def funct(one, two, three ):
    print( one, two, three )

def test(lol , argument_list):
    lol( **argument_list )

def gen (n):
    for i in range(n):
        yield i
gg = gen(10)

