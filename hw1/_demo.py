import numpy as np

class CAA:

    def __init__(self):
        pass

    @staticmethod
    def a():
        print("hello world")

    def b(self):
        self.a()

caa = CAA()
caa.b()












A = np.array([
    np.array([[1, 1], [2, 2]]),
    np.array([[3, 3]]),
    np.array([[4, 4], [5, 5], [6, 6]])
])

A_2 = -1 * np.array([
    np.array([[1, 1], [2, 2]]),
    np.array([[3, 3]]),
    np.array([[4, 4], [5, 5], [6, 6]])
])
print([np.column_stack((A[i], A_2[i])) for i in range(len(A))])

B = np.array([
    [1, 1], [2, 2],
    [3, 3],
    [4, 4], [5, 5], [6, 6]
])

B_2 = -1 * np.array([
    [1, 1], [2, 2],
    [3, 3],
    [4, 4], [5, 5], [6, 6]
])
print(np.column_stack((B, B_2)))



# B = np.array([
#     [1, 2],
#     [2, 3],
#     [3, 4, 5]
# ])
#
# C = np.row_stack([A, B])
# print(C)
# print("----")
# print(C[0])
# print("----")
#
# print(C[1])
#
# np.save("tmp.npy", [A, B])
