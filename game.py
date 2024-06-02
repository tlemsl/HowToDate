

"""
1: 첫인사(그림)
2: 카톡 대화(영화 고르기)
3: 식당 고르기
4: 손잡기
5: 대화하기(스크롤)
6: 싸움
7: 얘기 들어주기(드래그)
8: 쓰레기 치우기 1
9: 쓰레기 치우기 2
10: 엔딩

"""


def Game(state, action, answer, reward):
    next_state=0
    done=False
    if state==1:
        if action==1 or action==2 or action==3:
            next_state=2
            reward+=1
            if action==1:

                answer+=1
        else:
            next_state=1
            reward-=1
    elif state==2:
        if action==4 or action==2 or action==3:
            next_state=3
            reward+=1
            if action==4:
                answer+=1
        else:
            next_state=2
            reward-=1
    elif state==3:
        if action==4 or action==2 or action==3:
            next_state=4
            reward+=1
            if action==4:
                answer+=1
        else:
            next_state=3
            reward-=1
    elif state==4:
        if action==1 or action==2 or action==3:
            next_state=5
            reward+=1
            if action==3:
                answer+=1
        else:
            next_state=4
            reward-=1
    elif state==5:
        if action==4 or action==2 or action==3:
            next_state=6
            reward+=1
            if action==4:
                answer+=1
        else:
            next_state=5
            reward-=1
    elif state==6:
        if action==4 or action==2 or action==3:
            next_state=7
            reward+=1
            if action==4:
                answer+=1
        else:
            next_state=6
            reward-=1
    elif state==7:
        if action==1 or action==2 or action==3:
            next_state=8
            reward+=1
            if action==3:
                answer+=1
        else:
            next_state=7
            reward-=1

    elif state==8:
        if action==1 or action==2 or action==3:
            next_state=9
            reward+=1
            if action==3:
                answer+=1

        else:
            next_state=8
            reward-=1
    elif state==9:
        if action==1 or action==2 or action==3:
            next_state=10
            reward+=1
            if action==3:
                answer+=1

        else:
            next_state=9
            reward-=1
    else:
        done=True
        print(f"answer{answer}")
        if answer>=4:
            reward+=20
        else:
            reward+=10

    return next_state, answer, reward, done
