import numpy as np


def get_outlier_bound(nums):
    Q1 = np.percentile(nums, 25)
    Q3 = np.percentile(nums, 75)
    IQR = Q3 - Q1

    # Determine outliers
    outlier_lower_bound = Q1 - 1.5 * IQR
    outlier_upper_bound = Q3 + 1.5 * IQR

    return outlier_lower_bound, outlier_upper_bound


def ask_continue():
    while True:
        user_input = input("\n더 질문하기 (y/n): ")
        if user_input.lower() == "y":
            return True
        elif user_input.lower() == "n":
            return False
        else:
            print("잘못된 입력입니다. 'y'를 입력하여 계속하거나 'n'을 입력하여 중지하세요.")
