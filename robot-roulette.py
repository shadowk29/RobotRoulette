import numpy as np
import random
from collections import OrderedDict
import inspect
import math
import types

__version__='1.0'

    
class RouletteBot:
    def __init__(self, func):
        self.func = func
        self.hp = 100
        self.history = []

    def guess(self, e_history, ties, alive, start):
        num = self.func(self.hp, e_history, ties, alive, start)
        try:
            num = int(num)
        except TypeError:
            num = 0
            
        if num > self.hp or num < 0 or not type(num) == type(0):
            num = 0        
        return num

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
    bracket['SarcomaBotMk10-0'] = RouletteBot(sarcoma_bot_mk_ten)
    bracket['SarcomaBotMk10-1'] = RouletteBot(sarcoma_bot_mk_ten)
    bracket['SarcomaBotMk10-2'] = RouletteBot(sarcoma_bot_mk_ten)
    bracket['TENaciousBot'] = RouletteBot(TENacious_bot)
    bracket['SurvivalistBot'] = RouletteBot(SurvivalistBot)
    bracket['HalvsiestBot'] = RouletteBot(HalvsiesBot)
    bracket['GeometricBot'] = RouletteBot(geometric)
    bracket['BoxBot'] = RouletteBot(BoxBot)
    bracket['UpYoursBot'] = RouletteBot(UpYoursBot)
    bracket['AggroCalcBot'] = RouletteBot(aggresiveCalculatingBot)
    bracket['DeterminBot'] = RouletteBot(deterministicBot)
    bracket['AAAAUpYoursBot'] = RouletteBot(antiantiantiantiupyoursbot)
    bracket['GenericBot'] = RouletteBot(generic_bot)
    bracket['CoastBotV2'] = RouletteBot(coastV2)
    bracket['MehBot'] = RouletteBot(meh_bot)
    bracket['Meh20Bot'] = RouletteBot(meh_bot20)
    bracket['MehRanBot'] = RouletteBot(meh_ran)
    bracket['Bot13'] = RouletteBot(bot13)
    bracket['CautiousBot'] = RouletteBot(cautious_gambler)
    bracket['PercentBot'] = RouletteBot(percent)
    bracket['HalflifeS3Bot'] = RouletteBot(HalflifeS3)
    bracket['BloodBot'] = RouletteBot(blood_bot)
    bracket['MeanKickBot'] = RouletteBot(mean_kick)
    bracket['PolyBot'] = RouletteBot(polybot)
    bracket['PerfectFractionBot'] = RouletteBot(ThePerfectFraction)
    bracket['CautiousGamblerBot2'] = RouletteBot(cautious_gambler2)
    bracket['KickbanBot'] = RouletteBot(kickban)
    bracket['AntiKickBot'] = RouletteBot(antiKickBot)
    bracket['SquareUpBot'] = RouletteBot(squareUp)
    bracket['SnetchBot'] = RouletteBot(snetchBot)
    bracket['BoundedRandomBot'] = RouletteBot(boundedRandomBot)
    bracket['AggressiveBoundedRandomBotV2'] = RouletteBot(aggressiveBoundedRandomBotV2)
    bracket['MataHari2Bot'] = RouletteBot(MataHari2Bot)
    bracket['OgBot'] = RouletteBot(ogbot)
    bracket['BandaidBot'] = RouletteBot(BandaidBot)
    bracket['GetAlongBot'] = RouletteBot(GetAlongBot)
    bracket['StrikerBot'] = RouletteBot(strikerbot)
    bracket['HardCodedBot'] = RouletteBot(hc)
    bracket['OverfittedBot'] = RouletteBot(OverfittedBot)
    bracket['SmartBot'] = RouletteBot(smart_bot)
    return bracket

def tournament_score(score):
    tscore = dict()
    for key in score.keys():
        tscore[key] = score[key][0] + 0.5*score[key][1]
    return sorted(tscore.items(), key=lambda x:x[1], reverse=True)
       
def main():
    bracket = reset_bracket()
    rounds = int(np.ceil(np.log2(len(bracket))))
    round_eliminated = {key: np.zeros(rounds, dtype=np.int64) for key in list(bracket.keys())}
    score = {key: [0,0] for key in list(bracket.keys())}
    N = 100000
    for n in range(N):
        if n%100 == 0:
            print n
        winner, tied, eliminated = tournament(bracket)
        if not tied:
            score[winner][0] += 1
        else:
            score[winner[0]][1] += 1
            score[winner[1]][1] += 1
        for key in list(eliminated.keys()):
            round_eliminated[key][eliminated[key]] += 1
        bracket = reset_bracket()
    tscore = tournament_score(score)

    avg = 0
    for val in tscore:
        avg += val[1]
    avg /= len(tscore) * float(N)
    print 'Average Tournament Score: {0:.2g}'.format(avg)

    i = 0
    print 'Name\tScore\tWinRate\tTieRate\tElimination Probability'
    for key, val in tscore:
        i += 1
        print '{5}. {0}\t{1:.5f}\t{2:.2f}%\t{3:.2f}%\t{4}%'.format(key, val/float(N), 100*(score[key][0]/float(N)), 100*(score[key][1]/float(N)), np.around(round_eliminated[key]/float(N)*100,0).astype(np.int64), i)

    
def tournament(bracket):
    unused = bracket
    eliminated = {}
    used = {}
    start = len(unused)
    roundnum = 0
    while len(unused) + len(used) > 1:
        alive = len(unused) + len(used)
        #print 'Contestants remaining: {0}'.format(len(unused) + len(used))
        if len(unused) == 1:
            index = list(unused.keys())[0]
            used[index] = unused[index]
            unused = used
            used = {}
            roundnum += 1
        elif len(unused) == 0:
            unused = used
            used = {}
            roundnum += 1
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
                eliminated[blueindex] = roundnum
                red.hp -= rednum
                red.history.append(rednum)
                if red.hp > 0:
                    used[redindex] = red
                    del unused[redindex]
                else:
                    del unused[redindex]
                    eliminated[redindex] = roundnum
            elif bluenum > rednum:
                #print 'Red dies!'
                del unused[redindex]
                eliminated[redindex] = roundnum
                blue.hp -= bluenum
                blue.history.append(bluenum)
                if blue.hp > 0:
                    used[blueindex] = blue
                    del unused[blueindex]
                else:
                    del unused[blueindex]
                    eliminated[blueindex] = roundnum
            else: #if you're still tied at this point, both die
                #print 'Both die!'
                del unused[redindex]
                del unused[blueindex]
                eliminated[redindex] = roundnum
                eliminated[blueindex] = roundnum
    if unused:
        return list(unused.keys())[0], False, eliminated
    elif used:
        return list(used.keys())[0], False, eliminated
    else:
        return [redindex, blueindex], True, eliminated
        
                
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

    if hp == 100 and alive == 2:
        return hp - 1


    #This part is taken from Survivalist Bot, thanks @SSight3!
    remaining = alive - 2
    btf = 0

    rt = remaining
    while rt > 1:
        rt = float(rt / 2)
        btf += 1

    if ties > 2:
        return hp - 1

    if history:
        opp_hp = 100 - sum(history)

        #This part is taken from Geometric Bot, thanks @Mnemonic!

        fractions = []
        health = 100
        for x in history:
            fractions.append(float(x) / health)
            health -= x

        #Modified part

        if len(fractions) > 1:
            i = 0
            ct = True
            while i < len(fractions)-1:
                if abs((fractions[i] * 100) - (fractions[i + 1] * 100)) < 1:
                    ct = False
                i += 1

            if ct:
                expected = fractions[i] * opp_hp
                return expected

        if alive == 2:
            if hp > opp_hp:
                return hp - 1
            return hp
        if hp > opp_hp + 1:
            if opp_hp <= 15:
                return opp_hp + 1
            if ties == 2:
                return opp_hp + 1
            else:
                return opp_hp
    else:
        n = 300 // (alive - 1) + 1 #greater than
        if n >= hp:
            n = hp - 1
        return n

def halfpunch(hp, history, ties, alive, start): #revisited
    punch = hp - 1
    if alive == 2:
        return punch
    if history:
        if hp > 1:
            punch = np.ceil(hp/2.05) + ties + np.floor(ties / 2)
        else:
            punch = 1
    else:
        punch = 42 + ties + np.floor(ties / 2)
    if punch >= hp:
        punch = hp - 1
    return punch

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



def spitballBot(hp, history, ties, alive, start):
    # Spitball a good guess                                                                                                           
    roundsLeft = np.ceil(np.log2(alive)) # Thanks @Heiteira!                                                                     
    divFactor = roundsLeft**.8
    base = ((hp-1) / divFactor) + 1.5 * ties
    value = np.floor(base)

    # Don't bid under 20                                                                                                              
    if value < 20:
        value = 20 # Thanks @KBriggs!                                                                                                 

    # Don't bet over the opponent's HP                                                                                                 
    # (It's not necessary)                                                                                                            
    opponentHp = 100
    for h in history:
        opponentHp -= h

    if value > opponentHp:
        value = opponentHp

    # Always bet less than your current HP                                                                                            
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
    mult=3
    gang = False
    if history:
            count = 0
            for bid in history:
                    if bid % mult == 0:
                            count += 1
            if count == len(history):
                    gang = True
    if gang and hp<100:#Both bots need to have a history for a handshake
            if hp > 100-sum(history):
                    a=np.random.randint(0,hp/9+1)
            elif hp == 100-sum(history):
                    a=np.random.randint(0,hp/18+1)
            else:
                    return 1
            return a*mult
    elif gang:
            fS = (100-sum(history))/mult
            return (fS+1)*mult
    else:
            fP = hp/mult
            answer = fP*mult
            opp_hp = 100-sum(history)
            if history:
                    if len(history)>1:
                            opp_at_1 = 100-history[0]
                            ratio = 1.0*history[1]/opp_at_1
                            guessedBet= ratio*opp_hp
                            answer = np.ceil(guessedBet)+1
                    else:
                            if 1.0*hp/opp_hp>1:
                                    fS = opp_hp/mult
                                    answer = fS*mult
            else:
                    fS = hp/(2*mult)
                    answer = fS*mult+mult*2 +np.random.randint(-1,1)*3
            if answer > hp or alive == 2 or answer < 0:
                    if alive == 2 and hp<opp_hp:
                      answer = hp
                    else:
                      answer = hp-1
            if hp > 1.5*opp_hp:
                    return opp_hp + ties
            if ties:
              answer += np.random.randint(2)*3
            return answer

def guess_bot(hp, history, ties, alive, start):
   enemy_hp = 100 - sum(history)
   if len(history) == 1:
       if history[0] == 99:
           return 2
       else:
           return 26 + ties*2

   elif len(history) > 1:
       next_bet_guess = sum(history)//(len(history)**2)
       if alive == 2: 
           return hp
       elif alive > 2: 
           if hp > next_bet_guess + 1:
               return (next_bet_guess + 1 + ties*2)
           else:
               return (hp - 1)

   else:
       #Thank you Sarcoma bot. See you in Valhalla.
       startBid = hp / 3
       maxAdditionalBid = np.round(hp * 0.06) if hp * 0.06 > 3 else 3
       additionalBid = np.random.randint(2, maxAdditionalBid)
       return int(startBid + additionalBid + ties)

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
    return hp - 1 + ties
  opp_hp = 100 - sum(history)
  spend = 35 + np.random.randint(0, 11)
  if history:
    spend = min(spend, history[-1] + np.random.randint(0, 5))
  frugal = min(int((hp * 5. / 8) + ties), hp)
  return min(spend, opp_hp, frugal)

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


def sarcomaBotMkTwo(hp, history, ties, alive, start):
    if inspect.stack()[1][3] != 'guess' and inspect.stack()[1] == 5:
        return hp
    if alive == 2:
        return hp - 1
    if not history:
        startBid = hp / 2
        maxAdditionalBid = np.round(hp * 0.125) if hp * 0.125 > 2 else 2
        additionalBid = np.random.randint(1, maxAdditionalBid)
        return int(startBid + additionalBid + ties)
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp:
        return opponentHealth + ties
    minimum = np.round(hp * 0.6)
    maximum = hp - 1 or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1


def deterministicBot(hp, history, ties, alive, start):
    if alive == 2:
        return (hp - 1 + ties)
    if hp == 100:
        return 52
    if hp == 48:
        return 26
    if hp == 22:
        return 13
    if hp == 9:
        return 6
    if hp == 3:
        return 2
    else:
        return hp


def aggresiveCalculatingBot(hp, history, ties, alive, start):
    opponentsHP = 100 - sum(history)
    if opponentsHP == 100: # Get past the first round
        return int(min(51+ties, hp-1+ties))
    if alive == 2: # 1v1
        return hp - 1 + ties
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
            return int(min(theoreticalBet, hp - 1)) # no point suiciding
    maxRoundsLeft = np.ceil(np.log2(alive))
    theoreticalBet = hp / float(maxRoundsLeft)
    additionalRandomness = round(np.random.random()*maxRoundsLeft*2) 
    # want to save something for the future
    actualBet = min(theoreticalBet + additionalRandomness + ties, hp - 2)
    return int(actualBet)

def sarcomaBotMkFour(hp, history, ties, alive, start):
    def isSafe(parentCall):
        frame, filename, line_number, function_name, lines, index = parentCall
        if function_name is not 'guess':
            return False
        if line_number > 60:
            return False
        return True

    if not isSafe(inspect.stack()[1]):
        return hp
    if alive == 2:
        return hp - 1
    if not history:
        startBid = hp / 2
        maxAdditionalBid = np.round(hp * 0.08) if hp * 0.08 > 2 else 2
        additionalBid = np.random.randint(1, maxAdditionalBid)
        return int(startBid + additionalBid + ties)
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp:
        return opponentHealth + ties
    minimum = np.round(hp * 0.55)
    maximum = np.round(hp * 0.80) or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1

def sarcomaBotMkSix(hp, history, ties, alive, start):
    def isSafe(parentCall):
        frame, filename, line_number, function_name, lines, index = parentCall
        if function_name is not 'guess':
            return False
        if line_number > 60:
            return False
        return True

    if not isSafe(inspect.stack()[1]):
        return hp
    if alive == 2:
        return hp - 1
    if not history:
        startBid = hp / 2
        maxAdditionalBid = np.round(hp * 0.06) if hp * 0.06 > 3 else 3
        additionalBid = np.random.randint(2, maxAdditionalBid)
        return int(startBid + additionalBid + ties)
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp:
        return opponentHealth + ties
    minimum = np.round(hp * 0.55)
    maximum = np.round(hp * 0.70) or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1

def sarcomaBotMkSeven(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if not history:
        return 30 + ties
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp * 0.50:
        return opponentHealth + ties
    minimum = np.round(hp * 0.54)
    maximum = np.round(hp * 0.58) or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1

def antiantiantiantiupyoursbot(hp, history, ties, alive, start):
  def stuck():
    return [0, ('Whoops!', 'I', 'accidentally', 'replaced', 'your', 'code!')]
  def stick():
    return [0, ("Line", "number", 16, "guess", "it's", "faked :)")]
  inspect.stack =  stick
  spend = min(sarcomaBotMkSix(hp, history, ties, alive, start), hp)
  if not history:
    spend = 40 + np.random.randint(0, 10)
  inspect.stack = stuck
  return spend


def generic_bot(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if not history:
        return int(hp * 7.0 / 13)
    opp = 100 - sum(history)
    if opp < hp:
        return opp + ties
    max_sac = np.maximum(int(hp * 0.7), 1)
    rate = history[-1] * 1.0 / (history[-1] + opp)
    return int(np.minimum(max_sac, rate * opp + 1))

def coast(hp, history, ties, alive, start):
   if alive == 2:
   # Last round, go all out
       return hp - 1 + ties
   else:
       # Find the next power of two after the starting number of players
       players = start
       while math.log(players, 2) % 1 != 0:
         players += 1

       # This is the number of total rounds
       rounds = int(math.log(players, 2))

       bid = 99 / rounds

       if alive == start:
           # First round, add our leftover hp to this bid to increase our chances
           leftovers = 99 - (bid * rounds)
           return bid + leftovers
       else:
           # Else, just try and coast

           opp_hp = 100 - sum(history)
           # If opponent's hp is low enough, we can save some hp for the 
           # final round by bidding their hp + 1
           return min(bid, opp_hp + 1)

        
def coastV2(hp, history, ties, alive, start):
   # A version of coast bot that will be more agressive in the early rounds

   if alive == 2:
   # Last round, go all out
       return hp - 1 + ties
   else:
       # Find the next power of two after the starting number of players
       players = start
       while math.log(players, 2) % 1 != 0:
         players += 1

       # This is the number of total rounds
       rounds = int(math.log(players, 2))

       #Decrease repeated bid by 2 to give us more to bid on the first 2 rounds
       bid = (99 / rounds) - 2

       if len(history) == 0:
           # First round, add 2/3rds our leftover hp to this bid to increase our chances
           leftovers = 99 - (bid * rounds)
           return int(bid + math.ceil(leftovers * 2.0 / 3.0))
       elif len(history) == 1:
           # Second round, add 1/3rd of our leftover hp to this bid to increase our chances
           leftovers = 99 - (bid * rounds)
           return int(bid + math.ceil(leftovers * 1.0 / 3.0))
       else:
           # Else, just try and coast

           opp_hp = 100 - sum(history)
           # If opponent's hp is low enough, we can save some hp for the 
           # final round by bidding their hp + 1
           return int(min(bid, opp_hp + 1))


def cautious_gambler(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if(history):
        opp_hp = 100 - sum(history)
        remaining_rounds = np.ceil(np.log2(start)) - len(history)

        start_bet = opp_hp / 2
        buff = int((hp - start_bet)/remaining_rounds if remaining_rounds > 0 else (hp - start_bet)) 
        buff_bet = np.random.randint(0, buff) if buff > 0 else 0
        bet = start_bet + buff_bet + ties

        if bet >= hp or bet > opp_hp:
            bet = np.minimum(hp - 1, opp_hp)

        return int(bet)
    else:
        start_bet = hp / 2
        rng_bet = np.random.randint(3,6)

        return int(start_bet + rng_bet + ties)


def meh_bot(hp, history, ties, alive, start):
    # Attempt one      MehBot         | 0.020 | 1.6%    | 0.8%    | [34 36 12 10  6  1]%
    # Attempt two      MehBot         | 0.106 | 10.1%   | 0.8%    | [60  6  7  8  8  2]%
    point = hp / 2 + 3

    if ties > 1:
        ties += 1

    # Go all out on last round
    if alive == 2:
        return hp - 1

    opponent_hp = 100 - sum(history)

    if hp < 3:
        return 1
    elif not history:
        # Start with 30, This will increase the chance of dying first round but hopefully better fighting chance after
        return 30 + ties
    elif point > opponent_hp:
        # Never use more points then needed to win
        return opponent_hp + ties
    elif point >= hp:
        return hp - 1
    else:
        return point


def bot13(hp, history, ties, alive, start):
    win = 100 - sum(history) + ties
    #print "Win HP: %d" % win
    if alive == 2:
        #print "Last round - all in %d" % hp
        return hp - 1
    elif hp > win:
        #print "Sure win"
        return win
    #print "Don't try too hard"
    return 13 + ties

def sarcomaBotMkNine(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if not history:
        return 30 + np.random.randint(0, 4) + ties
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp * 0.50:
        return opponentHealth + ties
    minimum = np.round(hp * 0.54)
    maximum = np.round(hp * 0.58) or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1

def sarcomaBotMkEight(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if not history:
        return 30 + np.random.randint(0, 2) + ties
    opponentHealth = 100 - sum(history)
    if opponentHealth < hp * 0.50:
        return opponentHealth + ties
    minimum = np.round(hp * 0.54)
    maximum = np.round(hp * 0.58) or 1
    return np.random.randint(minimum, maximum) if minimum < maximum else 1

def percent(hp, history, ties, alive, start):
    if len(history) == 0:
        #First round, roundon low bid
        return int(random.randint(10,33))
    elif alive == 2:
        #Last round, go all out
        return int(hp - 1 + ties)
    else:
        # Try and calculate the opponents next bid by seeing what % of their hp they bid each round
        percents = []
        for i in range(0, len(history)):
            hp_that_round = 100 - sum(history[:i])
            hp_spent_that_round = history[i]
            percent_spent_that_round = 100.0 * (float(hp_spent_that_round) / float(hp_that_round)) 
            percents.append(percent_spent_that_round)

        # We guess that our opponents next bid will be the same % of their current hp as usual, so we bid 1 higher.
        mean_percent_spend = sum(percents) / len(percents)
        op_hp_now = 100 - sum(history)
        op_next_bid = (mean_percent_spend / 100) * op_hp_now
        our_bid = op_next_bid + 1

        #print mean_percent_spend
        #print op_hp_now
        #print op_next_bid

        # If our opponent is weaker than our predicted bid, just bid their hp + ties
        if op_hp_now < our_bid:
            return int(op_hp_now + ties)
        elif our_bid >= hp:
            # If our bid would kill us, we're doomed, throw a hail mary
            return int(random.randint(1, hp))
        else:
            return int(our_bid + ties)

        
def meh_ran(hp, history, ties, alive, start):
    # Attempt one      MehBot         | 0.020 | 1.6%    | 0.8%    | [34 36 12 10  6  1]%
    # Attempt two      MehBot         | 0.106 | 10.1%   | 0.8%    | [60  6  7  8  8  2]%
    # Attempt three    MehBot         | 0.095 | 9.1 %   | 0.7 %   | [70  3  5  6  6  0]%

    point = hp / 2 + 3
    if ties > 1:
        ties += 1
    # Go all out on last round
    if alive == 2:
        return hp - 1
    opponent_hp = 100 - sum(history)
    if hp < 3:
        return 1
    elif not history:
        # randome number between 33
        return random.randint(33, 45)
    elif point > opponent_hp:
        # Never use more points then needed to win
        return opponent_hp + ties
    elif point >= hp:
        return hp - 1
    else:
        return point

def meh_bot20(hp, history, ties, alive, start):
    # Attempt one      MehBot         | 0.020 | 1.6%    | 0.8%    | [34 36 12 10  6  1]%
    # Attempt two      MehBot         | 0.106 | 10.1%   | 0.8%    | [60  6  7  8  8  2]%
    point = hp / 2 + 3
    opponent_hp = 100 - sum(history)

    percents = []
    for i in range(0, len(history)):
        hp_that_round = 100 - sum(history[:i])
        hp_spent_that_round = history[i]
        percent_spent_that_round = 100.0 * (float(hp_spent_that_round) / float(hp_that_round))
        percents.append(percent_spent_that_round)

    try:
        opp_percent_point = opponent_hp * (max(percents) / 100)
    except:
        opp_percent_point = 100

    if ties > 1:
        ties += 1
    # Go all out on last round
    if alive == 2:
        return hp - 1

    if hp < 3:
        return 1
    elif not history:
        # randome number between 33
        return random.randint(33, 45)
    elif len(history) > 3:
        if point > opponent_hp:
            return min(opponent_hp + ties, opp_percent_point + ties)
    elif point > opponent_hp:
        # Never use more points then needed to win
        return opponent_hp + ties
    elif point >= hp:
        return hp - 1
    else:
        return point

def HalflifeS3(hp, history, ties, alive, start):
    ''' Bet a half of oponent life + 2 '''
    if history:
        op_HP = 100 - sum(history)
        return np.minimum(hp-1, np.around(op_HP/2) + 2 + np.floor(1.5 * ties) )
    else:
        return hp/3

def blood_bot(hp, history, ties, alive, start):
    enemy_hp = 100 - sum(history)
    if history:
        if len(history) == 1:
            if history[0] == 99:
                return 2

        if alive == 2:
            return hp

        if enemy_hp <= 5:
            return enemy_hp - 2 + ties*2

        if enemy_hp <= 10:
            return enemy_hp - 5 + ties*2

        if (hp - enemy_hp) > 50:
            return (2*enemy_hp/3 + ties*4)

        if (hp - enemy_hp) > 20:
            return (2*enemy_hp/3 + ties*3)

        if (hp - enemy_hp) < 0:
            #die gracefully
            return hp - 1 + ties

    else:
        startBid = hp / 3
        maxAdditionalBid = np.round(hp * 0.06) if hp * 0.06 > 3 else 3
        additionalBid = np.random.randint(2, maxAdditionalBid)
        return int(startBid + additionalBid + ties)

def mean_kick(hp, history, ties, alive, start):
    if alive == 2:
        return hp-1

    if not history:
        return 35

    opp_hp = 100 - sum(history)
    if opp_hp*2 <= hp:
        return opp_hp + ties
    else:
        return min(round(opp_hp/2) + 3 + ties*2, hp-1 + (ties>0))

def MataHari2Bot(hp, history, ties, alive, start):
    ''' 
    Round 1: Determine what our opponents do in the beginning with no information,
    give the assumed best counter.

    Round 2: Using the information from round 1, determine which of our opponents
    are possible given their history.  Determine what they'll do with our history
    (which we know from our hp) and give the assumed best counter.

    Round 3+: Same as round 2, except we have to make up our history based on our
    hp (not knowing our own history is a really odd decision but w/e), which leads
    to less and less precision.
    '''

    this = MataHari2Bot
    INITIAL_HP = 100

    # It's not that I hate Ogbot, it's that otherwise we get recursion
    if inspect.currentframe().f_back.f_back.f_code.co_name != 'tournament':
        return 1

    def simulate(method, hp, history, ties, alive, start):
        # print("Running simulation for %s when we are %i HP, opponent is %i HP" % (method.__name__, hp, INITIAL_HP - np.sum(history) + 1))
        return [ method(hp, history, ties, alive, start) for x in range(200) ]

    def best_guess(hp, results):
        low = np.min(results)
        high = np.max(results)
        if high == low or high - low <= 5:
            guess = high
        else:
            guess = np.median(results) 
        if guess is None:
            return 1
        return guess

    class BotInfo:
        def __init__(self, name, method, hp, history, ties, alive, start):
            self.name = name
            self.method = method
            self.round1_simulations = simulate(self.method, hp, history, ties, alive, start)
            self.round1_guess = best_guess(hp, self.round1_simulations)
            self.guesses = { }

        def guess(self, hp, history, ties, alive, start):
            opponent_hp = INITIAL_HP - np.sum(history)
            key = (hp * 100) + opponent_hp
            if key not in self.guesses:
                try:
                    simulations = simulate(self.method, opponent_hp, [ INITIAL_HP - hp ], ties, alive, start)
                except ValueError:
                    # BoundedRandomBot has a problem if you feed it a hp of 1 and a history of 99; this doesn't
                    # occur normally because it's not actually included in reset_brackets, perhaps by oversight?
                    # Anyways, ignore it.
                    # print("Error occurred calling %s with (%i, %s, %i, %i, %i); our hp %i, opponent history %s" % (self.name, opponent_hp, [ INITIAL_HP - hp ], ties, alive, start, hp, history))
                    if self.name != 'BoundedRandomBot':
                        raise
                    simulations = [ 1 ]

                self.guesses[key] = best_guess(hp, simulations)
            return self.guesses[key]

    if not hasattr(this, 'bots'):
        # Could've used reset_brackets, but abiding by the rules to not touch RouletteBot even if they're not the ones
        # being used in the tournament
        print("Initializing %s; this should only happen once" % (this.__name__))
        # We have to skip OgBot or the "Simulate OgBot 200 times" then leads into OgBot calling every other bot and performance just goes
        # to hell
        blacklist = [ this.__name__, 'ogbot' ]
        this.bots = [ ]
        for key, value in globals().iteritems():
            if key not in blacklist and type(value) == type(this) and len(inspect.getargspec(value).args) == 5:
                bot = BotInfo(key, value, hp, history, ties, alive, start) 
                # Ignore the suicide bots.  Even if we beat them in round 1, we'll be down too many hp to win, so they
                # will just skew our medians pointless
                # Also ignore the wildly random bots
                if bot.round1_guess < 70 and (np.max(bot.round1_simulations) - np.min(bot.round1_simulations)) < 40:
                    this.bots.append(bot)

    # Round1 will always be the same, no need to recalculate every time
    if len(history) == 0:
        if hasattr(this, 'round1_guess'):
            return this.round1_guess + ties
        guesses = [ bot.round1_guess for bot in this.bots ]
    else:
        # Otherwise, determine which bots could have given us the round1 guess we have in history
        # Weight this; the guess of a bot that deterministically returns this number is worth more
        # than one that returned it only 5% of the time
        guesses = [ ]
        for bot in this.bots:               
            if history[0] in bot.round1_simulations:
                guess = bot.guess(hp, history, ties, alive, start)
                guesses.extend( [ guess ] * bot.round1_simulations.count(history[0]))

    if len(guesses) == 0:
        # yolo
        guess = mean_kick(hp, history, ties, alive, start) + 1
    else:
        guess = int(math.ceil(np.median(guesses))) + 1
        # As long as it doesn't cost us too much, keep going up.  This catches the various
        # "do it like Xbot but adding 1" bots
        while guess in guesses:
            guess = guess + 1
    guess = min(guess, hp - 1 + ties, INITIAL_HP - np.sum(history) + ties)
    if len(history) == 0:
        this.round1_guess = guess
        print("MataHari2Bot's Round1 Guess: %i" % (this.round1_guess))
        #ordered_bots = sorted(this.bots, key = lambda x: x.round1_guess)
        #i = 1
        #for bot in ordered_bots:
        #   print("\t%i: %s (%i)" % (i, bot.name, bot.round1_guess))
        #   i = i + 1
    return guess

def ogbot(hp, history, ties, alive, start):
    otherBots = functions = [f for f in globals().values() if type(f) == types.FunctionType]

    def whatWouldOtherbotDo(bot,hp,history,ties,alive,start):
        botname = bot.__name__
        # got to avoid self referencing
        if(botname)=='ogbot':
            return -999

        # avoid non bot functions 
        try:
            return bot(hp,history,ties,alive,start)
        except (TypeError, NameError, RuntimeError):
            return -999

    otherBotCurrentBids = []

    otherBotOpeningBids = []    

    for bot in otherBots:
        otherBotCurrentBids.append(whatWouldOtherbotDo(bot,hp,history,ties,alive,start))
        otherBotOpeningBids.append(whatWouldOtherbotDo(bot,100,[],0,start,start))

    # remove invalid outputs
    otherBotCurrentBids = filter(lambda a: a != -999, otherBotCurrentBids)
    otherBotOpeningBids = filter(lambda a: a != -999, otherBotOpeningBids)

    # if this is the second round or later, try to deal with bots who were likely to pass the first round
    if len(history)>ties:
        medianOpening = np.median(otherBotOpeningBids)
        # get the bots who probably passed the first round and aren't committing suicide
        likelySuccessBots = [(a,b) for a,b in zip(otherBotCurrentBids, otherBotOpeningBids) if (b>medianOpening) & (a<hp)]      
        otherBotCurrentBids = [a for a,b in likelySuccessBots]


    out = min(np.median(otherBotCurrentBids) + 1,hp-1)
    return out

def polybot(hp, history, ties, alive, start):
  opp_hp = 100 - sum(history)
  if alive == 2:
    return hp - 1
  round = len(history)
  spend = 0
  if round == 0:
    spend = 35 + np.random.randint(1, 11)
  elif round <= 2:
    spend = int(history[-1] * 2 / (4 - round)) + np.random.randint(5 * round - 4, 10 * round - 5)
  else:
    poly = np.polyfit(xrange(0, round), history, 2)
    spend = int(np.polyval(poly, round)) + np.random.randint(1, 4)
    spend = max(spend, opp_hp / 2 + 3)
  return min(spend, hp - 1, opp_hp) 


def sarcoma_bot_mk_ten(hp, history, ties, alive, start):
    def bid_between(low, high, hp, tie_breaker):
        minimum = np.round(hp * low)
        maximum = np.round(hp * high) or 1
        return np.random.randint(minimum, maximum) + tie_breaker if minimum < maximum else 1

    if alive == 2:
        return hp - 1 + ties
    current_round = len(history) + 1
    tie_breaker = (ties * ties) + 1 if ties else ties
    if current_round == 1:
        return 39 + tie_breaker
    opponent_hp = 100 - sum(history)
    if opponent_hp < hp * 0.50:
        return opponent_hp + ties
    if current_round == 2:
        return bid_between(0.45, 0.50, hp, tie_breaker)
    if current_round == 3:
        return bid_between(0.50, 0.55, hp, tie_breaker)
    if current_round == 4:
        return bid_between(0.55, 0.60, hp, tie_breaker)
    if current_round == 5:
        bid_between(0.60, 0.65, hp, tie_breaker)
    return hp - 1 + ties


def cautious_gambler2(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    if(history):
        opp_hp = 100 - sum(history)
        remaining_rounds = np.ceil(np.log2(start)) - len(history)

        start_bet = opp_hp / 2
        buff = int((hp - start_bet)/remaining_rounds if remaining_rounds > 0 else (hp - start_bet)) 
        buff_bet = np.random.randint(0, buff) if buff > 0 else 0
        bet = start_bet + buff_bet + ties

        if bet >= hp or bet > opp_hp:
            bet = np.minimum(hp - 1, opp_hp)

        return int(bet)
    else:
        start_bet = hp * 0.35
        rng_bet = np.random.randint(3,6)

        return int(start_bet + rng_bet + ties)


def kickban(hp, history, ties, alive, start):
    if alive == 2:
        return hp-1

    if not history:
        return 36

    if history[0]==35:
        somean = 1
    else:
        somean = 0

    return min(mean_kick(hp, history, ties, alive, start) + somean*3, hp-1)

def antiKickBot(hp, history, ties, alive, start):
    if alive == 2:
        return (hp - 1 + ties)
    amount = np.ceil((float(hp) / 2) + 1.5)
    opponentsHP = 100 - sum(history)
    amount = min(amount, opponentsHP) + ties
    return amount

def squareUp(hp, history, ties, alive, start):

    #Taken from Geometric Bot
    opponentHP = 100 - sum(history)

    # Need to add case for 1
    if hp == 1:
        return 1

    # Last of the last - give it your all
    if alive == 2:
        if ties == 2 or opponentHP < hp-1:
            return hp - 1

    #Calculate your bet (x^(4/5)) with some variance
    myBet = np.maximum(hp - np.power(hp, 4./5), np.power(hp, 4./5))
    myBet += np.random.randint(int(-hp * 0.05) or -1, int(hp * 0.05) or 1);
    myBet = np.ceil(myBet)
    if myBet < 1:
        myBet = 1
    elif myBet >= hp:
        myBet = hp-1
    else:
        myBet = int(myBet)

    #If total annihilation is a better option, dewit
    if opponentHP < myBet:
        if ties == 2:
            return opponentHP + 1
        else:
            return opponentHP

    #If the fraction is proven, then outbid it (Thanks again, Geometric bot)
    if history and history[0] != history[-1]:
        health = 100
        fraction = float(history[0]) / health
        for i,x in enumerate(history):
            newFraction = float(x) / health
            if newFraction + 0.012*i < fraction or newFraction - 0.012*i > fraction:
                return myBet
            health -= x
        return int(np.ceil(opponentHP * fraction)) + 1    
    else:
        return myBet

def snetchBot(hp, history, ties, alive, start):
    if alive == 2:
        return hp-1

    opponent_hp = 100
    history_fractions = []
    if history:
        for i in history:
            history_fractions.append(float(i)/opponent_hp)
            opponent_hp -= i
        if opponent_hp <= hp/2:
            #print "Squashing a weakling!"
            return opponent_hp + (ties+1)/3

        average_fraction = float(sum(history_fractions)) / len(history_fractions)
        if history_fractions[-1] < average_fraction:
            #print "Opponent not raising, go with average fraction"
            next_fraction = average_fraction
        else:
            #print "Opponent raising!"
            next_fraction = 2*history_fractions[-1] - average_fraction
        bet = np.ceil(opponent_hp*next_fraction) + 1
    else:
        #print "First turn, randomish"
        bet = np.random.randint(35,55)

    if bet > opponent_hp:
        bet = opponent_hp + (ties+1)/3
    final_result = bet + 3*ties
    if bet >= hp:
        #print "Too much to bet"
        bet = hp-1
    return final_result

def boundedRandomBot(hp, history, ties, alive, start):
    max_possible_bid = hp - 1
    if alive == 2 or max_possible_bid == 0:
        return max_possible_bid
    
    if history:
        opp_hp = 100 - sum(history)
        bid_ceiling = min(opp_hp+1, max_possible_bid)
    else:
        bid_ceiling = max_possible_bid
    return np.random.randint(1, bid_ceiling+1)

def aggressiveBoundedRandomBotV2(hp, history, ties, alive, start):
    max_possible_bid = hp - 1
    if alive == 2 or max_possible_bid == 0:
        return max_possible_bid

    if history:
        opp_hp = 100 - sum(history)
    else:        
        opp_hp = 100

    bid_floor = min(np.ceil(opp_hp * 0.5), max_possible_bid)
    return np.random.randint(bid_floor, max_possible_bid+1)

def BandaidBot(hp, history, ties, alive, start):
    if alive == 2:
        return hp-1

    if history:
        opp_hp = 100 - sum(history)
        opp_last_hp = 100 - sum(history[:-1])

        if history[-1] <= opp_last_hp / 3:
            return 1 + np.random.randint(0, ties or 1) 
        elif history[-1] > opp_last_hp / 2:
            return min(opp_hp - 1, hp)
        else:
            if history[-1] < hp/2:
                return np.random.randint(history[-1], hp/2)
            else:
                return np.floor(hp/2)
    else:
        return np.floor(hp/3)

def GetAlongBot(hp, history, ties, alive, start):
    if alive == 2:
        return hp-1

    if history:
        opp_hp = 100 - sum(history)
        opp_last_hp = 100 - sum(history[:-1])
        count = 0
        for i in range(0, len(history)):
            hp_that_round = 100 - sum(history[:i])
            hp_spent_that_round = history[i]
            if hp_that_round / 3 - 1 <= hp_spent_that_round <= hp_that_round / 2:
                count += 1
        if count == len(history): #It's probably BandaidBot!
            return 2
        else:
            return min(opp_hp - 1, np.floor(hp/3))
    else:
        return np.floor(hp/3)

def strikerbot(hp, history, ties, alive, start):
    #get our magic number (tm) for useful things
    def magic_number(num):
        return np.floor(num / 2)
    #get opponent's hp and round number
    opp_hp = 100 - sum(history)
    round = 1
    if history:
        round = len(history) + 1
    #set strike initial value, by default it's all out
    strike = hp - 1
    #let 'er rip if last round
    if alive == 2:
        return strike
    if history:
        if hp > 1:
            #strike with a special calculation, using magic number shenanigans
            strike = np.ceil(hp/(2.045 + (magic_number(round) / 250)) ) + 1 + ties + magic_number(ties)
        else:
            #fallback
            strike = 1
    else:
        #round 1 damage
        strike = 42 + ties ** 2
    if opp_hp <= strike:
        #if opponent is weaker than strike then don't waste hp
        strike = opp_hp + ties
    if strike >= hp:
        #validations galore
        strike = hp - 1
    return strike

def OverfittedBot(hp, history, ties, alive, start):
    if alive == 2:
        return hp - 1
    opponent_hp = 100 - np.sum(history)
    if opponent_hp == 100:
        guess = 40 + ties
    elif len(history) == 1 and history[0] == 39:
        guess = opponent_hp * 0.50 + 1 + ties
    else:
        guess = mean_kick(hp, history, ties, alive, start) + 1 
    return min(guess, hp - 1 + ties, 100 - np.sum(history) + ties)


def hc(hp, history, ties, alive, start):
    plan = [40,27,20,10,2,5,5,5,5,5,2,2,2,2,2,1,1,1,1,1,100,1]
    ohp = 100 - sum(history)
    if ohp < hp * 0.16:
        return ohp + 1
    x = 100
    k = 0
    while x > hp:
        x -= plan[k]
        k += 1
    if plan[k] > ohp:
        return ohp + 1
    return plan[k] + ties

def ThePerfectFraction(hp, history, ties, alive, start):
    thePerfectFraction = 7 * hp / 13

    if alive == 2:
        return hp - 1

    opponent_hp = 100 - sum(history)

    if not history:
        # Need to up our game to overcome the kickers
        return 42 + ties
    if thePerfectFraction > opponent_hp:
        return opponent_hp + ties

    return thePerfectFraction + 1 + ties

def smart_bot(hp, history, ties, alive, start):
    op_high_bid = 0
    op_hp = 100
    hptoret = 1
    his = False
    tienum = (ties * ties) + 1 if ties else ties
    if not history:
        return 42 + tienum
        #return obots(hp, history, ties, alive, start)
    if his:
        if history[0]==36:
            return min(mean_kick(hp, history, ties, alive, start) + 3, hp-1)

    if alive == 2:
        if history[0]==36:
            return max(mean_kick(hp, history, ties, alive, start) + 3, hp-1)
        return hp - 1
    if history:
        his = True

    if his:
        for h in history:
            if int(h) > op_high_bid:
                op_high_bid = h
    if his:
        op_hp = 100 - sum(history)
        if op_hp >= hp:
            return np.floor(hp / 3)


    roundn = len(history) + 1
    #if roundn > 6:
        #return hp - 1
    """Overrideable(If above did not execute of course)"""

    if op_hp < hp:
        """Kick em hard"""
        hptoret = mean_kick(hp, history, ties, alive, start)
    return hptoret


if __name__=='__main__':
    main()
