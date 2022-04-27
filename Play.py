from othello.OGame import OthelloGame
from othello.bots.Random import BOT

game=OthelloGame(n=8)

class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))

game.play(black=Human(), white=BOT())

