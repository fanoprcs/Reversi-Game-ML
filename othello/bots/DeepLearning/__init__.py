import numpy as np
import random
from othello.OthelloUtil import getValidMoves
from othello.bots.DeepLearning.OthModel import OthelloModel
from othello.OGame import OthelloGame

class BOT():
    games = 0
    times = 0
    def __init__(self, board_size, *args, **kargs):
        self.board_size=board_size
        self.model = OthelloModel( input_shape=(self.board_size, self.board_size) )
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')
            pass
        
        self.collect_gaming_data=False
        self.history=[]
    
    def getAction(self, game, color):
        weights = self.__getweights(game, color, self.__initWeight())#根據戰局變化出現不同的權重
        predict = self.model.predict(game)
        valid_positions=getValidMoves(game, color)
        valids=np.zeros((game.size), dtype='int')
        valids[ [i[0]*self.board_size+i[1] for i in valid_positions] ]=1
        for i in range(self.board_size*self.board_size):
            if valids[i] == 1:
                predict[i]+=0.01
            else:
                predict[i]=0
        predict*=weights
        pos_backup = -1#記錄下第一次的選擇
        for i in range(self.board_size*self.board_size):
            
            position = self.randomPredict(predict)
            if position == -1:
                #print('pos == -1')
                position = pos_backup
                break
            copygame = game.copy()
            tmp = (position//self.board_size, position%self.board_size)
            if i == 0:#如果每一步棋都不好那就選最開始的那步
                pos_backup = position
            self.executeMove(copygame, color, tmp)
            opp_move = self.getOpponent(copygame, -color)#如果這步棋結束後對手能夠占角落, 則重新思考
            
            if opp_move == (0, 0) or opp_move == (7, 0) or opp_move == (0, 7) or opp_move == (7, 7):
                #print('he got coner')
                predict[position] = 0
            else:
                break
        if self.collect_gaming_data:
            tmp=np.zeros_like(predict)
            tmp[position]=1.0
            self.history.append([np.array(game.copy()), tmp, color])
        position=(position//self.board_size, position%self.board_size)
        #print(position)
        return position
    def getOpponent(self, game, color):#用自己的模型判斷對手是否會走什麼
        weights = self.__getweights(game, color, self.__initWeight())#根據戰局變化出現不同的權重
        predict = self.model.predict(game)
        valid_positions=getValidMoves(game, color)
        valids=np.zeros((game.size), dtype='int')
        valids[ [i[0]*self.board_size+i[1] for i in valid_positions] ]=1
        for i in range(self.board_size*self.board_size):
            if valids[i] == 1:
                predict[i]+=0.01
            else:
                predict[i]=0
        predict*=weights
        position = self.randomPredict(predict)
        position=(position//self.board_size, position%self.board_size)
        #print(position)
        return position
    def randomPredict(self, predict):#在一定範圍以內隨機取前兩名
        best = second = 0.0
        best_index = -1
        second_index = -1
        for i in range(self.board_size*self.board_size):
            if predict[i]>best:
                best = predict[i]
                best_index = i
        for i in range(self.board_size*self.board_size):
            if predict[i]>second and i!=best_index:
                second = predict[i]
                second_index = i
        #print(predict)
        #print(best_index)
        #print(second_index)
        if best == 0:
            return -1
        if second/best < 0.9:
            return best_index
        else:#第一跟第二差距在一定範圍內則接受隨機分配
            ran = random.randint(1,2)
            if ran == 1:
                return best_index
            else:
                return second_index
            
    def __getweights(self, game, color, weights):
        #如果角落是我的, 則更改週邊權重
        #避免被插隊卡點, 由四角往週邊依序蔓延, 蔓延不出去則不會更改權重
        ltop = topl = rtop = topr = ldown = rdown = downl = downr = 0 #避免延伸時把危險點重設為高權重時反而幫對面站到角落
        if game[0][0] == color:#左上
            weights[9] = 100 #不那麼重要但可以下
            alter = 0  #用來判斷角落旁的棋形是長怎樣
            for i in range(self.board_size-2):
                if game[i][0] == color:#row
                    if alter == 1: #如果變白又變黑表示對方成功卡點就不判斷了
                        break
                    weights[(i+1)*8] = 10000 - (i*1000) #戰局可能會多條併行, 距離越遠慢慢降低權重
                    ltop = 1 #如果是自己慢慢擴張的情況, 對方下了另一邊的危險棋且對邊角落還沒人, 由於被上行改權重, 也下了危險棋剛好把對面危險棋吃掉, 就相當於給對面角落, 很虧
                elif game[i][0] == -color:
                    alter = 1
                    weights[(i+1)*8] = 20000 + (i*20000) #距離越遠權重越高因為對方成功卡點的話損失更大
                    ltop = 0 #如果是翻一整排的對手旗子, 對方下了另一邊的危險棋, 給對方角落合理, 不然搶了角落反而被卡整排很虧, 所以不用在最後調整權重
                else:
                    break
            alter = 0
            for i in range(self.board_size-2):
                if game[0][i] == color: #col
                    if alter == 1:
                        break
                    weights[i+1] = 10000 - (i*1000)
                    topl = 1
                elif game[0][i] == -color:
                    alter = 1
                    weights[i+1] = 20000 + (i*20000)
                    topl = 0
                else:
                    break
            
        if game[0][7] == color: #右上
            weights[14] = 100
            alter = 0
            for i in range(self.board_size-2):
                if game[i][7] == color:
                    if alter == 1:
                        break
                    weights[(i+1)*8+7] = 10000 - (i*1000)
                    rtop = 1
                elif game[i][7] == -color:
                    alter = 1
                    weights[(i+1)*8+7] = 20000 + (i*20000)
                    rtop = 0
                else:
                    break
            alter = 0
            for i in range(self.board_size-2):
                if game[0][7-i] == color:
                    if alter == 1:
                        break
                    weights[6-i] = 10000 - (i*1000)
                    topr = 1
                elif game[0][7-i] == -color:
                    alter = 1
                    weights[6-i] = 20000 + (i*20000)
                    topr = 0
                else:
                    break
                    
        if game[7][0] == color: #左下
            weights[49] = 100
            alter = 0
            for i in range(self.board_size-2):
                if game[7-i][0] == color: #row
                    if alter == 1:
                        break
                    weights[(6-i)*8] = 10000 - (i*1000)
                    ldown = 1
                elif game[7-i][0] == -color:
                    alter = 1
                    weights[(6-i)*8] = 20000 + (i*20000)
                    ldown = 0
                else:
                    break
            alter = 0
            for i in range(self.board_size-2):
                if game[7][i] == color:
                    if alter == 1:
                        break
                    weights[57+i] = 10000 - (i*1000)
                    downl = 1
                elif game[7][i] == -color:
                    alter = 1
                    weights[57+i] = 20000 + (i*20000)
                    downl = 0
                else:
                    break
                    
        if game[7][7] == color: #右下
            weights[54] = 100
            alter = 0
            for i in range(self.board_size-2):
                if game[7-i][7] == color: #row
                    if alter == 1:
                        break
                    weights[(6-i)*8 + 7] = 10000 - (i*1000)
                    rdown = 1
                elif game[7-i][7] == -color:
                    alter = 1
                    weights[(6-i)*8 + 7] = 20000 + (i*20000)
                    rdown = 0
                else:
                    break
            alter = 0
            for i in range(self.board_size-2):
                if game[7][7-i] == color:
                    if alter == 1:
                        break
                    weights[62-i] = 10000 - (i*1000)
                    downr = 1
                elif game[7][7-i] == -color:
                    alter = 1
                    weights[62-i] = 20000 + (i*20000)
                    downr = 0
                else:
                    break
        #判斷邊邊是否可以吃
                    
        
        #預防犯的愚蠢錯誤
        if game[1][1] == -color and game[0][0] == 0:#左上
            weights[2] = weights[16] = 0.000001
            if topr == 1:
                weights[1] = 0.000001
            if ldown == 1:
                weights[8] = 0.000001
        if game[1][6] == -color and game[0][7] == 0:#右上
            weights[5] = weights[23] = 0.000001
            if topl == 1:
                weights[6] = 0.000001
            if rdown == 1:
                weights[15] = 0.000001
        if game[6][1] == -color and game[7][0] == 0:#左下
            weights[40] = weights[58] = 0.000001
            if ltop == 1:
                weights[48] = 0.000001
            if downr == 1:
                weights[57] = 0.000001
        if game[6][6] == -color and game[7][7] == 0:#右下
            weights[47] = weights[61] = 0.000001
            if rtop == 1:
                weights[55] = 0.000001
            if downl == 1:
                weights[62] = 0.000001
        return weights
    def __initWeight(self):#一般而言權重
        weights = [0] * self.board_size * self.board_size
        #四個角落
        weights[0] = weights[7] = weights[56] = weights[63] = 200000
        #左安全邊
        weights[16] = weights[40] = 2000
        weights[24] = weights[32] = 1000
        #上安全邊
        weights[2] = weights[5] = 2000
        weights[3] = weights[4] = 1000
        #右安全邊
        weights[23] = weights[47] = 2000
        weights[31] = weights[39] = 1000
        #下安全邊
        weights[58] = weights[61] = 2000
        weights[59] = weights[60] = 1000
        #左上危險點
        weights[1] = weights[8] = weights[9] = 0.000001
        
        #右上危險點
        weights[6] = weights[15] = weights[14] = 0.000001
       
        #左下危險點
        weights[48] = weights[57] = weights[49] = 0.000001
      
        #右下危險點
        weights[55] = weights[62] = weights[54] = 0.000001
    
        #中間安全點
        for i in range(self.board_size * self.board_size):
            if weights[i] == 0:
                weights[i] = 100
        return weights
    def self_play_train(self, args):
        print('train '+str(BOT.times))
        BOT.times+=1
        self.collect_gaming_data=True
        def gen_data():
            def getSymmetries(board, pi):
                # mirror, rotational
                pi_board = np.reshape(pi, (len(board), len(board)))
                l = []
                for i in range(1, 5):
                    for j in [True, False]:
                        newB = np.rot90(board, i)
                        newPi = np.rot90(pi_board, i)
                        if j:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        l += [( newB, list(newPi.ravel()) )]
                return l
            self.history=[]
            history=[]
            game=OthelloGame(self.board_size)
            game.play(self, self, verbose=args['verbose'])
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b,p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result=game.isEndGame()
            return [(x[0],x[1]) for x in history if (game_result==0 or x[2]==game_result)]
        
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            data+=gen_data()
        
        self.collect_gaming_data=False
        
        self.model.fit(data, batch_size = args['batch_size'], epochs = args['epochs'])
        self.model.save_weights()
        
    def executeMove(self, board, color, position):
        y, x = position
        board[y][x] = color
        for direction in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
            flips = []
            valid_route=False
            for size in range(1, len(board)):
                ydir = y + direction[1] * size
                xdir = x + direction[0] * size
                if xdir >= 0 and xdir < len(board) and ydir >= 0 and ydir < len(board):
                    if board[ydir][xdir]==-color:
                        flips.append((ydir, xdir))
                    elif board[ydir][xdir]==color:
                        if len(flips)>0:
                            valid_route=True
                        break
                    else:
                        break
                else:
                    break

            if valid_route:
                for i in range(len(flips)):
                    board[flips[i][0]][flips[i][1]]=color        