from AIGamePlatform import Othello
from othello.bots.DeepLearning import BOT

class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))
        
BOARD_SIZE=8
bot=BOT(board_size=BOARD_SIZE)
app=Othello() # 會開啟瀏覽器登入Google Account，目前只接受@mail1.ncnu.edu.tw及@mail.ncnu.edu.tw
 
@app.competition(competition_id='test')
def _callback_(board, color): # 函數名稱可以自訂，board是當前盤面，color代表黑子或白子
    return bot.getAction(board, color) # 回傳要落子的座標