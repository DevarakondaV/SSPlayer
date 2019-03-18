import cv2

"""
pre: game != None, img_fun != none
post: Creates window displaying an image
"""
def cv_disp(game, img_fun):
    """
    Function displays images
    args:
    img_fun: function which returns an image
    game: game instance
    """
    assert game is not None
    assert img_fun is not None and callable(img_fun) == 1
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)


    while True:
        img = img_fun()
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
