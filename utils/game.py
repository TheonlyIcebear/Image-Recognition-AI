import numpy as np

class Game:
    def __init__(self, grid=None):
        self.grid = np.array([ 
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]) if grid is None else np.array(np.split(np.array(grid), 3))

    @property
    def drawed(self):
        return not self.open_slots.size

    @property
    def open_slots(self):
        grid = self.grid.flatten()
        return np.where(grid == 0)[0]
    
    @staticmethod
    def _calculate_open_slots(grid):
        grid = grid.flatten()
        return np.where(grid == 0)[0]

    @staticmethod
    def _threats(grid, player, tiles):

        search = sorted(([0] * (3 - tiles)) + [player] * tiles)
        
        rows = [[x * 3 + tile for tile in np.where(grid[x, :] == 0)[0]] for x in range(3) if sorted(grid[x, :]) == search]
        cols = [[tile * 3 + y for tile in np.where(grid[:, y] == 0)[0]] for y in range(3) if sorted(grid[:, y]) == search]

        diag = np.diag(grid)
        flip = np.diag(np.fliplr(grid))

        if sorted(diag) == search:
            diag1 =  ((np.where(diag == 0)[0] * 3) + np.where(diag == 0)[0]).tolist()
        else:
            diag1 = []
        
        if sorted(flip) == search:
            diag2 = ((np.where(flip == 0)[0] * 3) + (2 - np.where(flip == 0)[0])).tolist()
        else:
            diag2 = []

        threats = sum(rows, []) + sum(cols, []) + diag1 + diag2

        return threats

    def eval(self, grid=None):

        if grid is None:
            grid = self.grid

        for player in [1, 2]:
            if any([ 
                (grid[row] == player).all() for row in range(3)
            ]): # Horizontal crosses
                return player

            elif any([ 
                (grid[:, column] == player).all() for column in range(3)
            ]): # Vertical crosses
                return player

            elif (np.diag(grid) == player).all(): # Downwards diagonal crosses
                return player

            elif (np.diag(np.fliplr(grid)) == player).all() :# Upwards diagonal crosses
                return player

        if not (grid == 0).any():
            return True

        return None

    def select(self, tile, player):
        index = int(tile)

        opponent = (1 - int(player)) + 1
        player = int(player) + 1
        
        grid = self.grid

        x = index // 3
        y = index % 3

        self.grid[x, y] = player

        win_threats =  self._threats(grid, player, 2)
        response = self.eval()
        

        if response is True: # If game is drawed
            return None

        return response == player

    # Find the optimal moves, using mini-max
    def best_moves(self, player): 
        grid = self.grid

        opponent = (1 - int(player)) + 1
        player = int(player) + 1

        lose_threats = self._threats(grid, opponent, 2)
        win_threats =  self._threats(grid, player, 2)

        opp_forks = self._threats(grid, opponent, 1)
        forks = self._threats(grid, player, 1)

        real_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1]
        opp_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1 and ((threat in win_threats) if win_threats else True) ]
        forks = [threat for threat in [*set(forks)] if forks.count(threat) > 1 and ((threat in lose_threats) if lose_threats else True)]

        best = self.open_slots

        if win_threats:
            best = win_threats

        elif lose_threats:
            best = lose_threats

        elif len(real_forks) > 1: # To defend multiple fork possibilities you need create a threat that blocks a fork

            stoppable = False
            found = False

            for tile in self.open_slots:
                grid = np.array(list(self.grid))

                old_opp_forks = len(real_forks)
                old_win_threats = len(win_threats)
                grid[tile // 3, tile % 3] = player

                new_opp_forks = self._threats(grid, opponent, 1)
                new_win_threats = self._threats(grid, player, 2)

                new_opp_forks = [*set([threat for threat in new_opp_forks if new_opp_forks.count(threat) > 1])]
                
                if len(new_opp_forks) < old_opp_forks and not found:
                    best = [tile]

                if (len(new_opp_forks) < old_opp_forks) and new_win_threats and (not new_win_threats[0] in new_opp_forks):
                    best.append(tile)
                    found = True

        elif  opp_forks:
            best = opp_forks

        elif 4 in self.open_slots:
            best = [4]

        elif (np.unique(self.grid).size == 2 and (opponent in self.grid)):
            opening = (np.where(self.grid == opponent)[0] * 3) + np.where(self.grid == opponent)[1]
            
            if (opening == 4):
                best = [0, 2, 6, 8]

            elif opening in [0, 2, 6, 8]:
                best = [4]

        return np.unique(best)
