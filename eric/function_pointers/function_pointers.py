def function_1(x, y):

    return x + y


def function_2(x, y):

    return x - y


def function_3(x, y):

    return x * y


def determine_function(flag):

    if flag == 'foo':
        f = function_1
    elif flag == 'bar':
        f = function_2
    elif flag == 'baz':
        f = function_3
    else:
        f = None

    return f

if __name__ == '__main__':

    for flag in ('foo', 'bar', 'baz'):
        f = determine_function(flag)
        print(f, f(5, 3))
