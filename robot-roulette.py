import numpy as np
import random
from collections import OrderedDict
__version__='0.1'

class RouletteBot:
    def __init__(self, func):
        self.func = func
        self.hp = 100
        self.history = []

    def guess(self, e_history, ties, alive, start):
        num = self.func(self.hp, e_history, ties, alive, start)
        if num > self.hp or num < 0:
            num = 0
        return int(num)

def reset_bracket():
    bracket = {}
    bracket['AverageBot'] = RouletteBot(average)
    bracket['LastBot'] = RouletteBot(last)
    bracket['RandomBot'] = RouletteBot(randombot)
    bracket['OneShotBot'] = RouletteBot(one_shot)
    bracket['OutBidBot'] = RouletteBot(outbid)
    bracket['PatheticBot'] = RouletteBot(pathetic_attempt_at_analytics_bot)
    bracket['HalfPunchBot'] = RouletteBot(halfpunch)
    bracket['KamikazeBot'] = RouletteBot(kamikaze)
    bracket['RobbieBot'] = RouletteBot(robbie_roulette)
    bracket['WorstCaseBot'] = RouletteBot(worst_case)
    bracket['SpitballBot'] = RouletteBot(spitballBot)
    bracket['AntiGangBot'] = RouletteBot(anti_gangbot)
    bracket['GangBot0'] = RouletteBot(gang_bot)
    bracket['GangBot1'] = RouletteBot(gang_bot)
    bracket['GangBot2'] = RouletteBot(gang_bot)
    bracket['GuessBot'] = RouletteBot(guess_bot)
    bracket['CalculatingBot'] = RouletteBot(calculatingBot)
    bracket['TitTatBot'] = RouletteBot(tatbot)
    bracket['SpreaderBot'] = RouletteBot(Spreader)
    bracket['KickBot'] = RouletteBot(kick)
    bracket['BinaryBot'] = RouletteBot(binaryBot)
    bracket['SarcomaBotMk2'] = RouletteBot(sarcomaBotMkTwo)
    bracket['TENaciousBot'] = RouletteBot(TENacious_bot)
    bracket['SurvivalistBot'] = RouletteBot(SurvivalistBot)
    bracket['HalvsiestBot'] = RouletteBot(HalvsiesBot)
    bracket['GeometricBot'] = RouletteBot(geometric)
    bracket['BoxBot'] = RouletteBot(BoxBot)
    bracket['UpYoursBot'] = RouletteBot(UpYoursBot)
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
            score[winner[0]][1] += 1
            score[winner[1]][1] += 1
        bracket = reset_bracket()
    tscore = tournament_score(score)
    print 'Name\t\tScore\tWinRate\tTieRate'
    for key, val in tscore:
        print '{0}:\t{1:.3f}\t{2:.1f}%\t{3:.1f}%\t'.format(key, val/float(N), 100*(score[key][0]/float(N)), 100*(score[key][1]/float(N)))

    print '<ol>'
    print '<b>Name\tScore\tWinRate\tTieRate</b>'
    for key, val in tscore:
        print '<li><b>{0}:</b>\t{1:.3f}\t{2:.1f}%\t{3:.1f}%</li>'.format(key, val/float(N), 100*(score[key][0]/float(N)), 100*(score[key][1]/float(N)))
    print '</ol>'

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

def randombot(hp, history, ties, alive, start):
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
    return np.minimum(hp - 1, hp - hp /(start - alive + 4) + ties * 2)

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

def anti_gangbot(hp, history, ties, alive, start):
    def round_to_seven(x):
        return int(7 * np.ceil(float(x)/7)) #Special function


    gang = False
    op_hp = 100
    if history:
        count = 0
        for bid in history:
            if bid % 7 == 0:
                count += 1
        if count > 1 or (len(history)==1 and count == 1):
            gang = True
        op_hp = 100-sum(history)

        if gang:                          # Anti-gangbot measures trump any value opponent bids
            if op_hp < 100:
                if op_hp > 100-hp:
                    return 3
                elif op_hp == 100-hp:
                    return 3
                else:
                    return 2
            else:
                fS = (100-hp)/7
                return int((fS+1)*7+1)
        else:
            if alive == 2:
                if hp > op_hp:
                    return hp - 1
                return hp
            if hp > op_hp + 1:
                if op_hp <= 15:
                    return op_hp # Can't lose much
                if ties > 0:
                    return hp # Just give up, kamikaze mode
                return op_hp + 1
            return op_hp
    else:
        n = round_to_seven(100 // (alive - 1) + 1) #greater than
        if n > 100:
            n = 98
        return n

def gang_bot(hp,history,ties,alive,start):

    tieBreak = 0
    if ties:
        tieBreak = np.random.randint(1,3)
        return tieBreak
    gang = False
    if history:
            count = 0
            for bid in history:
                    if bid % 7 == 0:
                            count += 1
            if count > 1 or (len(history)==1 and count == 1):
                    gang = True
    if gang and hp<100:#Both bots need to have a history for a handshake
            if hp > 100-sum(history):
                    return np.random.randint(0,hp/5 or 1)
            elif hp == 100-sum(history):
                    return np.random.randint(0,hp/10 or 1)
            else:
                    return 1
    elif gang:
            fS = (100-sum(history))/7
            return (fS+1)*7
    else:
            fP = hp/7
            answer = fP*7
            if history:
                    opp_hp = 100-sum(history)
                    if len(history)>1:
                            opp_at_1 = 100-history[0]
                            ratio = 1.0*history[1]/opp_at_1
                            guessedBet= ratio*opp_hp
                            answer = np.ceil(guessedBet)+1
                    else:
                            if 1.0*hp/opp_hp>1:
                                    fS = opp_hp/7
                                    answer = fS*7
            if answer > hp or alive == 2:
                    answer = hp-1
            if answer == 0:
                    floorSeven = hp/7
                    answer = floorSeven*7-7
            return answer

def guess_bot(hp, history, ties, alive, start):
   if len(history) == 1:
       if history[0] == 99:
           return 2
       if hp > ((history[0]/2) + 1):
           return ((history[0]/2) + 1)
       else:
           return ((hp/2) +2 )
   elif len(history) > 1:
       next_bet_guess = sum(history)//(len(history)**2)
       if alive == 2: 
           return hp - 1
       elif alive > 2: 
           if hp > next_bet_guess + 1:
               return (next_bet_guess + 1)
           else:
               return (hp - 1)

   else:
       #Trying to beat the half betters, won't beat the big guys
       #that bet by 2/3s hp
       return ((hp/2) + 1)

def calculatingBot(hp, history, ties, alive, start):
    opponentsHP = 100 - sum(history)
    if alive == 2: # 1v1
        if hp > opponentsHP: # we win!
            return hp - 1
        else: # hope for a tie
            return hp
    # Try to fit an exponential trendline and one up the trendline if it fits
    if len(history) >= 3: 
        xValues = range(1, len(history) + 1)
        # https://stackoverflow.com/a/3433503  Assume an exponential trendline
        coefficients = np.polyfit(xValues, np.log(history), 1, w = np.sqrt(history))
        def model(coefficients, x):
            return np.exp(coefficients[1]) * np.exp(coefficients[0] * x)
        yPredicted = [model(coefficients, x) for x in xValues]
        totalError = 0
        for i in range(len(history)):
            totalError += abs(yPredicted[i] - history[i])
        if totalError <= (len(history)): # we found a good fitting trendline
            # get the next predicted value and add 1
            theoreticalBet = np.ceil(model(coefficients, xValues[-1] + 1) + 1) 
            theoreticalBet += ties
            return max(theoreticalBet, hp - 1) # no point suiciding
    maxRoundsLeft = np.ceil(np.log2(alive))
    theoreticalBet = hp / float(maxRoundsLeft)
    additionalRandomness = round(np.random.random()*maxRoundsLeft) 
    # want to save something for the future
    actualBet = min(theoreticalBet + additionalRandomness + ties, hp - 2)
    return actualBet

def tatbot(hp, history, ties, alive, start):
  if alive == 2:
    return hp - 1
  opp_hp = 100 - sum(history)
  spend = 30 + np.random.randint(0, 21)
  if history:
    spend = min(spend, history[-1] + np.random.randint(0, 5))
  return min(spend, opp_hp, hp)

def Spreader(hp, history, ties, alive, start):
   if alive == 2:
       return hp-1
   if len(history) < 2:
       return hp/2
   return np.ceil(hp/alive)

def kick(hp, history, ties, alive, start):
    if alive == 2:
        return hp-1

    opp_hp = 100 - sum(history)
    if opp_hp*2 <= hp:
        return opp_hp + ties
    else:
        return min(round(opp_hp/2) + 1 + ties**2, hp-1 + (ties>0))
    
def binaryBot(hp, history, ties, alive, start):
    return int(np.floor(hp/2)) or 1

def sarcomaBot(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if not history:
        return int(hp / 2 + np.random.randint(0, hp / 4 or 1))
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp:
        return opponentHealth + 1
    minimum = np.round(hp * 0.75)
    maximum = hp - 1 or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1



def sarcomaBotMkTwo(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if not history:
        return int(hp / 2 + np.random.randint(0, hp / 8  or 1))
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp:
        return opponentHealth + 1
    minimum = np.round(hp * 0.60)
    maximum = hp - 1 or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1


def TENacious_bot(hp, history, ties, alive, start):
    max_amount=hp-(alive-1)*2;
    if max_amount<2: max_amount=2

    if alive==2: return hp-1
    if ties==0: return np.minimum(10, max_amount)
    if ties==1: return np.minimum(20, max_amount)
    if ties==2: return np.minimum(40, max_amount)
    # prevent function blowup
    return 2

def SurvivalistBot(hp, history, ties, alive, start):    

    #Work out the stats on the opponent
    Opponent_Remaining_HP = 100 - sum(history)
    Opponent_Average_Bid = Opponent_Remaining_HP

    if len(history) > 0:
        Opponent_Average_Bid = Opponent_Remaining_HP / float(len(history))


    HP_Difference = hp - Opponent_Remaining_HP

    #Work out the future stats on the others
    RemainingBots = (alive-2)
    BotsToFight = 0

    RemainderTree = RemainingBots

    #How many do we actually need to fight?
    while(RemainderTree > 1):
        RemainderTree = float(RemainderTree / 2)
        BotsToFight += 1

    #Now we have all that data, lets work out an optimal bidding strategy
    OptimalBid = 0
    AverageBid = 0

    #For some reason we've tied more than twice in a row, which means death occurs if we tie again
    #So better to win one round going 'all in'
    if ties > 1:
        if BotsToFight < 1:
            OptimalBid = hp - 1
        else:
            OptimalBid = hp - (BotsToFight+1)

        #Err likely we're 0 or 1 hp, so we just return our HP
        if OptimalBid < 1:
            return hp
        else:
            return OptimalBid

    #We have the upper hand (more HP than the opponent)
    if HP_Difference > 0:
        #Our first guess is to throw all of our opponent's HP at them
        OptimalBid = HP_Difference

        #But if we have more opponents to fight, we must divide our HP amongst our future opponents
        if BotsToFight > 0:
            #We could just divide our HP evenly amongst however many remaining bots there are
            AverageBid = OptimalBid / BotsToFight

            #But this is non-optimal as later bots will have progressively less HP
            HalfBid = OptimalBid / 2

            #We have fewer bots to fight, apply progressive
            if BotsToFight < 3:

                #Check it exceeds the bot's average
                if HalfBid > Opponent_Average_Bid:
                    return np.floor(HalfBid)
                else:
                    #It doesn't, lets maybe shuffle a few points over to increase our odds of winning
                    BidDifference = Opponent_Average_Bid - HalfBid

                    #Check we can actually match the difference first
                    if (HalfBid+BidDifference) < OptimalBid:
                        if BidDifference < 8:
                            #We add half the difference of the BidDifference to increase odds of winning
                            return np.floor(HalfBid + (BidDifference/2))
                        else:
                            #It's more than 8, skip this madness
                            return np.floor(HalfBid)

                    else:
                        #We can't match the difference, go ahead as planned
                        return np.floor(HalfBid)


            else:
                #There's a lot of bots to fight, either strategy is viable
                #So we use randomisation to throw them off!
                if bool(random.getrandbits(1)):
                    return np.floor(AverageBid)
                else:
                    return np.floor(HalfBid)

        else:
            #There are no other bots to fight! Punch it Chewy!
            return OptimalBid

    else:

        if hp == 100:
            #It appears to be our opening round (assumes opponent HP same as ours)
            #We have no way of knowing what our opponent will play into the battle

            #Only us in the fight? Full power to weapons!
            if BotsToFight < 1:
                return hp - 1
            else:
                #As what might happen is literally random
                #We will also be literally random
                #Within reason

                #Work out how many bots we need to pass
                HighestBid = hp - (BotsToFight+1)
                AverageBid = hp/BotsToFight
                LowestBid = np.floor(np.sqrt(AverageBid))

                #Randomly choose between picking a random number out of thin air
                #And an average
                if bool(random.getrandbits(1)):
                    return np.minimum(LowestBid,HighestBid)
                else:
                    return AverageBid

        else:
            #Oh dear, we have less HP than our opponent
            #We'll have to play it crazy to win this round (with the high probability we'll die next round)
            #We'll leave ourselves 1 hp (if we can)

            if BotsToFight < 1:
                OptimalBid = hp - 1
            else:
                OptimalBid = hp - (BotsToFight+1)

            #Err likely we're 0(???) or 1 hp, so we just return our HP
            if OptimalBid < 1:
                return hp
            else:
                return OptimalBid

def HalvsiesBot(hp, history, ties, alive, start):
    return np.floor(hp/2)          


def geometric(hp, history, ties, alive, start):
    opponentHP = 100 - sum(history)

    # If we're doomed, throw in the towel.
    if hp == 1:
        return 1

    # If this is the last battle or we can't outsmart the opponent, go all out.
    if alive == 2 or ties == 2:
        return hp - 1

    # If the opponent is weak, squish it.
    if opponentHP <= hp / 3:
        if ties == 2:
            return opponentHP + 1
        else:
            return opponentHP

    # If the opponent has full health, pick something and hope for the best.
    if not history:
        return np.random.randint(hp / 3, hp)

    # Assume the opponent is going with a constant fraction of remaining health.
    fractions = []
    health = 100
    for x in history:
        fractions.append(float(x) / health)
        health -= x
    avg = sum(fractions) / len(fractions)
    expected = int(avg * opponentHP)
    return min(expected + 2, hp - 1)

def BoxBot(hp, history, ties, alive, start):

    Opponent_HP = np.round(100 - sum(history))
    HalfLife = np.round(Opponent_HP/2)
    RandomOutbid = HalfLife + 1 + np.random.randint(0,HalfLife or 1)

    if hp < RandomOutbid:
        return hp - 1
    else:
        return RandomOutbid
def UpYoursBot(hp, history, ties, alive, start):
    willToLive = True# if "I" is in "VICTORY"

    args = [hp, history, ties, alive, start]
    enemyHealth = 100 - sum(history)
    roundNumber = len(history)

    if roundNumber is 0:
        # Steal HalfPunchBot
        return halfpunch(*args) + 2

    if alive == 2:
        # Nick OneShotBot
        return one_shot(*args)

    if enemyHealth >= hp:
        # Pinch SarcomaBotMkTwo
        return sarcomaBotMkTwo(*args) + 1

    if enemyHealth < hp:
        # Rip off KickBot
        return kick(*args) + 1

    if not willToLive:
        # Peculate KamikazeBot
        return kamikaze(*args) + 1

    
if __name__=='__main__':
    main()
