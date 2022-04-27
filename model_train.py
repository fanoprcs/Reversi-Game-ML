from othello.bots.DeepLearning import BOT

BOARD_SIZE=8
bot=BOT(board_size = BOARD_SIZE)

args={
    'num_of_generate_data_for_train': 4,
    'epochs': 5,
    'batch_size': 4,
    'verbose': True
}

iterations = 5000

for _ in range(iterations):
    bot.self_play_train(args)


