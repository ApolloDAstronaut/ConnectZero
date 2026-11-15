from game import *

board = init_board()
print(board)
current_player = 1
run = True
while run:
    player_title = 1 + (current_player == -1)
    board = place_piece(board, current_player, int(input(f"Player {player_title}: ")) - 1)
    print(board)
    if is_win(board, current_player):
        run = False
        print(f"Player {player_title} has won!")
        print('''HH   HH   OOOOO    OOOOO   RRRRRR    AAAAA   YY   YY
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA   YY YY
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA    YYY
HH   HH  OO   OO  OO   OO  RRRRRR   AAAAAAA     Y
HHHHHHH  OO   OO  OO   OO  RR RR    AA   AA     Y
HH   HH  OO   OO  OO   OO  RR  RR   AA   AA     Y
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA     Y
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA     Y
HH   HH   OO OO    OO OO   RR   RR  AA   AA     Y
HH   HH    OOO      OOO    RR   RR  AA   AA     Y''')
    current_player *= -1