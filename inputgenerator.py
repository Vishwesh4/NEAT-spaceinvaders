import cv2
import numpy as np
import time

# Number of inputs generated from this function is 76
def inputgen(test_img):

    start = time.time()
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    h,w = img.shape #210,160

    #Specifics of game, using priors for faster implementation and increase the frame rate
    starting_pixel = 36 #starting y coordinates of aliens
    gap_between_aliens = 18  #pixel height in between center of alients
    horizontal_gap_between_aliens = 16
    down_movement = 10 #Down movement of aliens
    self_y = 192 # the y-coordinate of self
    search_x_start = 22 #the lowest x-coordinate where alien can be found
    search_x_end = 139 #the largest x-coordinate where aliens can be found
    alien_species = 6 #No: of different rows of aliens
    color_aliens = 122 #grayscale color of alien
    color_self  =  98 #grayscale color of self
    bullet = 142 #grayscale color of bullet
    bullet_search = 20 #keeping search limited to [self-bullet_search,self+bullet_search]
    alien_size = 8 #the width of alien

    #Self position
    pos, = np.where(img[self_y]==color_self)
    if len(pos)==0:
        self_x = 0
    else:
        self_x = pos[(len(pos)//2)]

    #Get x,y coordinates of all aliens
    flag = 1
    iter = 0
    position = []
    vertical_pos  = []
    while flag:
        search_index = starting_pixel + 10*iter
        if (search_index)<self_y+10:
            pos, = np.where(img[search_index,search_x_start:search_x_end]==color_aliens)
        else:
            pos = []
            break
        if(len(pos)==0):
            iter+=1
        else:
            vertical_pos.append(search_index)
            position.append(pos+22)

            aliens=0
            while ((aliens+1)*gap_between_aliens+search_index)<self_y+10:
                gap = (aliens+1)*gap_between_aliens
                pos, = np.where(img[search_index+gap,search_x_start:search_x_end]==color_aliens)
                aliens+=1
                if(len(pos)==0):
                    iter+=1
                else:
                    vertical_pos.append(search_index+gap)
                    position.append(pos+22)
            flag = 0

    #Extract useful position out of Postion list
    Alien_locations = []
    for i in range(len(position)):
        X = position[i]
        flag = 1
        index = 0
        while flag:
            pos, = np.where( X[index:]<=X[index]+alien_size)
            temp = X[index:][pos]
            alien_coord = (temp[-1]+temp[0])//2
            Alien_locations.append([alien_coord,vertical_pos[i]])
            index = index + len(pos)
            if index>=len(X):
                flag = 0

    #Filling the dead aliens at the end as 0,0
    Enemies_killed = 36-len(Alien_locations)
    for i in range(Enemies_killed):
        Alien_locations.append([0,0])

    Alien_locations = (np.ravel((np.matrix(Alien_locations)).flatten())).tolist()

    #Extract bullet position in vicinity of the self around bullet_search
    search_left = max(self_x-bullet_search,search_x_start)
    search_right = min(self_x + bullet_search,search_x_end)
    bullet_search_area = test_img[starting_pixel:self_y,search_left:search_right]
    #Only consider the closest bullet which is the imminent threat
    temp_bullet = np.argwhere(bullet_search_area==bullet)
    if len(temp_bullet)==0:
        bullet_x,bullet_y = (0,0)
    else:
        nearest_bullet = np.argmax(temp_bullet[:,0])
        bullet_x,bullet_y = temp_bullet[nearest_bullet,1]+search_left,temp_bullet[nearest_bullet,0]+starting_pixel

    #Input to neural nets- self_x,alien_pos_x,alien_pos_y,bullet_x,bullet_y
    Input =  [self_x] + Alien_locations + [bullet_x,bullet_y] + [Enemies_killed]
    end = time.time()

    return Input
#Testing purpose
if __name__=='__main__':
    test_img = cv2.imread('../images/observation'+str(150)+'.png')
    # parameters (priors)
    # starting_pixel = 114
    starting_pixel = 114 #The bot is only required to see a part of the screen
    self_y = 192
    search_x_start = 22
    search_x_end = 139
    input = inputgen(test_img)
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    #For coloring
    x,y = input[-3],input[-2]
    loc = np.where(test_img[:,x]==107)
    if len(loc[0])==0:
        test_img[y:,x] = 200
    else:
        test_img[y:loc[0][0],x] = 200

    ob = test_img[starting_pixel:self_y+2,search_x_start-5:search_x_end+5]
    h_ob,w_ob = ob.shape #184,127
    inx = int(h_ob/3)
    iny = int(w_ob/3)
    ob = cv2.resize(ob, (iny, inx))
    ob = np.reshape(ob, (inx,iny))
    imgarray = np.ndarray.flatten(ob)
    print(len(imgarray))
    cv2.imshow('test_img2',ob)
    cv2.waitKey(0)
