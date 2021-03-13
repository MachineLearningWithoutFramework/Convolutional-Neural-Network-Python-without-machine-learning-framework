import numpy as np
import cv2
from model import *



def main():
    a1=cv2.imread("four.png",0)
    a2=cv2.imread("three.png",0)
    a3=cv2.imread("zero.png",0)
    test_data=[]
    test_data.append(a1);test_data.append(a2);test_data.append(a3)
    w,aw,r=run_model("digits.png","divide picture",test_data,(20,20))


if __name__ == "__main__":
    main()



