test = (1, 2, 3)
tester = iter(test)

for i in range(10):
    try:
        nextItem = next(tester)
        print(nextItem)
    except Exception as e:
        print(str(e))
        tester = iter(test)
        nextItem = next(tester)
        print(nextItem)
