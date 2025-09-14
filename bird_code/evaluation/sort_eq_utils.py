

def tuple_eq(r1, r2, eps):
    if len(r1) != len(r2):
        return False
    for c1, c2 in zip(r1, r2):
        if c1 is None and c2 is None:
            pass
        elif c1 is not None and c2 is not None:
            if isinstance(c1, float) and isinstance(c2, float):
                if abs(c1 - c2) < eps:
                    pass
                else:
                    return False
            elif c1 == c2:
                pass
            else:
                return False
        else:
            return False
    return True

def sort_unique(results, eps):
    results = list(results)
    results.sort(key=lambda r: tuple([(c is None, c) for c in r]))
    result_new = []
    for r in results:
        if len(result_new) == 0 or not tuple_eq(result_new[-1], r, eps):
            result_new.append(r)
    return result_new

def sort_eq(results1, results2, eps=1e-3):
    results1 = sort_unique(results1, eps)
    results2 = sort_unique(results2, eps)
    if len(results1) != len(results2):
        return False
    for r1, r2 in zip(results1, results2):
        if not tuple_eq(r1, r2, eps):
            return False
    return True


def test_sort_eq():
    assert sort_eq([], []) == True
    assert sort_eq([], [()]) == False
    assert sort_eq([()], []) == False
    assert sort_eq([()], [()]) == True

    assert sort_eq([(None, )], [(None, )]) == True
    assert sort_eq([(None, )], [()]) == False
    assert sort_eq([()], [(None, )]) == False

    assert sort_eq([(1, )], [(1, )]) == True
    assert sort_eq([(1, )], [(None, )]) == False
    assert sort_eq([(None, )], [(1, )]) == False

    assert sort_eq([(1., )], [(1.000000000000001, )]) == True
    assert sort_eq([(None, 1., )], [(None, 1.000000000000001, )]) == True
    assert sort_eq([(None, 1., )], [(None, None, )]) == False

    assert sort_eq([(None, None, )], [(None, None, ), (None, None, )]) == True
    assert sort_eq([(None, None, ), (None, None, )], [(None, None, )]) == True
    assert sort_eq([(None, None, ), (None, None, )], [(None, None, )]) == True

    assert sort_eq([('a', )], [('a', ), ('a', )]) == True
    assert sort_eq([('a', )], [('a', ), ('b', )]) == False

    assert sort_eq([(1.0, )], [(0.999999999999999999, ), (1.0000000000000001, )]) == True
    assert sort_eq([(1.0, ), (2.0, ), (1.99999999999999999999, )], [(0.999999999999999999, ), (2.0000000000000001, )]) == True

    assert sort_eq([(1.0, ), (2.0, 3.0)], [(0.999999999999999999, ), (2.0000000000000001, )]) == False
    assert sort_eq([(1.0, ), (None, 2.0)], [(0.999999999999999999, ), (2.0000000000000001, )]) == False

    assert sort_eq([('b', 2.000000000000001), ('a', 1.0), ('b', 1.9999999999999999)], [('b', 2.0), ('a', 1.000000000000001), ]) == True

    print('====== sort_eq: OK')


if __name__ == '__main__':
    test_sort_eq()