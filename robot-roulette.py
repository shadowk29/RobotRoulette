import numpy as np
from collections import OrderedDict

class RouletteBot:
    def __init__(self, func):
        self.func = func
        self.hp = 100
        self.history = []

    def guess(self, e_history, ties, alive, start):
        num = self.func(self.hp, e_history, ties, alive, start)
        if num > self.hp:
            num = 0
        return num

def reset_bracket():
    bracket = {}
    bracket['AverageBot'] = RouletteBot(average)
    bracket['LastBot'] = RouletteBot(last)
    bracket['RandomBot'] = RouletteBot(random)
    bracket['OneShotBot'] = RouletteBot(one_shot)
    bracket['OutBidBot'] = RouletteBot(outbid)
    bracket['PatheticBot'] = RouletteBot(pathetic_attempt_at_analytics_bot)
    bracket['PunchBot'] = RouletteBot(halfpunch)
    bracket['KamikazeBot'] = RouletteBot(kamikaze)
    bracket['RobbieBot'] = RouletteBot(robbie_roulette)
    bracket['WorstCaseBot'] = RouletteBot(worst_case)
    bracket['SpitballBot'] = RouletteBot(spitballBot)
    return bracket

def tournament_score(score):
    tscore = dict()
    for key in score.keys():
        tscore[key] = score[key][0] + 0.5*score[key][1]
    return sorted(tscore.items(), key=lambda x:x[1], reverse=True)
    
    
def main():
    bracket = reset_bracket()
    score = {key: [0,0] for key in list(bracket.keys())}
    N = 100000
    for n in range(N):
        winner, tied = tournament(bracket)
        if not tied:
            score[winner][0] += 1
        else:
            score[winner[0]][1] += .5
            score[winner[1]][1] += .5
        bracket = reset_bracket()
    tscore = tournament_score(score)
    for key, val in tscore:
        print key, val/float(N)

def tournament(bracket):
    unused = bracket
    used = {}
    start = len(unused)
    while len(unused) + len(used) > 1:
        alive = len(unused) + len(used)
        #print 'Contestants remaining: {0}'.format(len(unused) + len(used))
        if len(unused) == 1:
            index = list(unused.keys())[0]
            used[index] = unused[index]
            unused = used
            used = {}
        elif len(unused) == 0:
            unused = used
            used = {}
        else:
            
            redindex = np.random.choice(list(unused.keys()))
            blueindex = np.random.choice(list(unused.keys()))
            while blueindex == redindex:
                blueindex = np.random.choice(list(unused.keys()))

            
            red = unused[redindex]
            blue = unused[blueindex]
            #print '{0}/{2} vs {1}/{3}'.format(redindex, blueindex, red.hp, blue.hp)
            ties = 0
            rednum = red.guess(blue.history, ties, alive, start)
            bluenum = blue.guess(red.history, ties, alive, start)
            #print 'Red: {0}/{2} vs Blue: {1}/{3}'.format(rednum, bluenum, red.hp, blue.hp)
            while rednum == bluenum:
                #print 'Red: {0} vs Blue: {1}'.format(rednum, bluenum, red.hp, blue.hp)
                ties += 1
                if ties == 3:
                    break
                rednum = red.guess(blue.history, ties, alive, start)
                bluenum = blue.guess(red.history, ties, alive, start)
            if rednum > bluenum:
                #print 'Blue dies!'
                del unused[blueindex]
                red.hp -= rednum
                red.history.append(rednum)
                if red.hp > 0:
                    used[redindex] = red
                    del unused[redindex]
                else:
                    del unused[redindex]
            elif bluenum > rednum:
                #print 'Red dies!'
                del unused[redindex]
                blue.hp -= bluenum
                blue.history.append(bluenum)
                if blue.hp > 0:
                    used[blueindex] = blue
                    del unused[blueindex]
                else:
                    del unused[blueindex]
            else: #if you're still tied at this point, both die
                #print 'Both die!'
                del unused[redindex]
                del unused[blueindex]
    if unused:
        return list(unused.keys())[0], False
    elif used:
        return list(used.keys())[0], False
    else:
        return [redindex, blueindex], True
        
                
def last(hp, history, ties, alive, start):
    if history:
        return 1 + np.minimum(hp-1, history[-1])
    else:
        return hp/3 + np.random.randint(-2,3)
    
def average(hp, history, ties, alive, start):
    if history:
        num = np.minimum(hp-1, int(np.average(history))+1)
    else:
        num = hp/3 + np.random.randint(-2,3)
    return num

def random(hp, history, ties, alive, start):
    return 1 + np.random.randint(0, hp)

def kamikaze(hp, history, ties, alive, start):
      return hp

def one_shot(hp, history, ties, alive, start):
      if hp == 1:
          return 1
      else:
          return hp - 1

def outbid(hp, history, ties, alive, start):
    if history:
        return np.minimum(hp-1,99-sum(history))
    if hp == 1:
        return 1
    return np.random.randint(hp/5, hp/2)

def pathetic_attempt_at_analytics_bot(hp, history, ties, alive, start):
    '''Not a good bot'''
    if history:
        opp_hp = 100 - sum(history)
        if alive == 2:
            if hp > opp_hp:
                return hp - 1
            return hp
        if hp > opp_hp + 1:
            if opp_hp <= 15:
                return opp_hp +1
            if ties > 0:
                return hp #Just give up, kamikaze mode
            return opp_hp + 1
        return opp_hp
    else:
        n = 300 // (alive - 1) + 1 #greater than
        if n >= hp:
            n = hp - 1
        return n

def halfpunch(hp, history, ties, alive, start):
    if hp > 1:
        return np.ceil(hp/2)
    else:
        return 1

def robbie_roulette(hp, history, ties, alive, start):
     if history:
         #If the enemy bot has a history, and it's used the same value every time, outbid that value
         if len(set(history)) == 1:
             return history[0] + 1
         #Else, average the enemy bot's history, and bid one more than the average
         else:
             return (sum(history) / len(history) + 1)
     #Else, return half of remaining hp
     else:
         return hp / 2

def worst_case(hp, history, ties, alive, start):
    return np.minimum(hp - 1, hp - hp /(start + 2 - alive) + ties * 2)

def BoundedRandomBot(hp, history, ties, alive, start):
    return np.ceil(max(np.random.randint(min(hp/3, 0.8*(100-sum(history))), 0.8*(100-sum(history))), hp-1, 1))


def spitballBot(hp, history, ties, alive, start):
    base = ((hp-1) / (alive-1)) + 1.5 * ties
    value = np.floor(base)

    if value < 10:
        value = 10

    if value >= hp:
        value = hp-1

    return value


if __name__=='__main__':
    main()
