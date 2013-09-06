import numpy
from numpy import *
from os import getcwd
from time import time
import jinja2
import webbrowser

# dimension 0 = round (0 to r-1)
# dimension 1 = home/away (0 to 1)
# dimension 2 = pitch (0 to p-1)

def make_group_fixtures(teams, pitches):
    result = numpy.zeros(((teams - 1) * teams / 2, 2, pitches), dtype = int16)
    c = 0
    for i in range(1, teams+1):
        for j in range(i+1, teams+1):
            result[c, 0, 0] = i
            result[c, 1, 0] = j
            c+=1
    return result

def make_all_fixtures(groupings, pitches, reverse=False):
    bump_number = 0
    result = zeros((0,2,pitches), dtype = int16)
    for g in groupings:
        fixtures = make_group_fixtures(g, pitches)
        fixtures = bump_team_numbers(fixtures, bump_number)
        result = vstack((result, fixtures))
        bump_number += g
    if reverse: result = concatenate((result, result), 0)
    random.shuffle(result)
    return result

def bump_team_numbers(fixtures, bump_value):
    fixtures[fixtures.nonzero()] += bump_value
    return fixtures

def gen_fixtures_closed(teams, groups, reverse=False):
    base = gen_group_closed(teams)
    result = base
    for i in range(1, groups):
        result = concatenate((result, base + (teams * i)), axis=2)
    if reverse: result = concatenate((result, result), 0)
    return result

def gen_group_closed(teams):
    if teams % 2 != 0:
        teams += 1
        remove = [teams]
    else:
        remove = None
    result = numpy.zeros((teams - 1, 2, teams / 2), dtype = int16)
    fixed = 1
    rotation = range(2, teams + 1)
    for j in range(teams - 1):
        result[j] = round_write(fixed, rotation)
        rotation = shift_rotation(rotation)
    if remove:
        result = remove_teams(result, remove)
        result = align_left(result)
        result = result[:, :, :-1]
    return result

def round_write(fixed, rotation):
    s = (len(rotation) + 1) / 2
    result = numpy.zeros((2, s), dtype=int16)
    result[0][0] = fixed
    for i in range(1, s):
        result[0][i] = rotation[i - 1]
    for i in range(s, 2 * s):
        result[1][(2 *s) - i - 1] = rotation[i - 1]
    return result

def shift_rotation(rotation):
    return rotation[-1:] + rotation[:-1]

def remove_teams(fixtures, removals):
    for r in fixtures:
        for p in range(size(fixtures, 2)):
            if r[0][p] in removals or r[1][p] in removals:
                r[0][p] = r[1][p] = 0
    return fixtures

def gen_fixtures(teams, groups, pitches, reverse=False, max_time = 15, round_target = 0):
    time_target = time() + max_time
    team_groupings = calc_groupings(teams, groups)
    if round_target == 0:
        round_target = minimum_rounds(team_groupings, min(pitches, int(teams / 2))) * [1, 2][reverse]
    if pitches * 2 == teams and teams % (2 * groups) == 0:
        fixtures = gen_fixtures_closed(teams / groups, groups, reverse=reverse)
    elif pitches >= int(teams / 2) and round_target == (max(team_groupings) - 1) * [1, 2][reverse]:
        fixtures = gen_fixtures_closed(max(team_groupings), len(team_groupings), reverse=reverse)
        removals = range(teams + 1, (max(team_groupings) * len(team_groupings)) + 1)
        fixtures = remove_teams(fixtures, removals)
        fixtures = align_left(fixtures)
        fixtures = fixtures[:, :, :pitches]
    else:
        fixtures = make_all_fixtures(team_groupings, pitches, reverse=reverse)
    best_fixtures = fixtures
    if pitches > (teams / 2):
        pitches = int(teams / 2)
        print "too many pitches - reducing to " + str(pitches)
    cur_rounds = size(fixtures, 0)
    test_row = cur_rounds - 1
    test_pitch = 0
    fit_row = 0
    times_permuted = 0
    print "aiming for " + str(round_target) + " rounds..."
    while cur_rounds > round_target:
        if time() > time_target:
            if cur_rounds < size(best_fixtures, 0): best_fixtures = fixtures
            print "timed out on " + str(size(best_fixtures, 0)) + " rounds, target is " + str(round_target) + "."
            return best_fixtures
        while not team_plays(fixtures, 0, fit_row):
            fit_row += 1
        team_one = fixtures[test_row, 0, test_pitch]
        team_two = fixtures[test_row, 1, test_pitch]
        if team_one == 0:
            test_row -= 1
            test_pitch = 0
            continue
        if fit_row >= test_row:
            if fit_row >= cur_rounds - 1:
                if times_permuted % 500 == 0 and times_permuted > 0: print str(times_permuted) + " iterations"
                fixtures, fail_flag = refit_matches(fixtures)
                if fail_flag or random.random()<0.2:
                    row, col = find_subarray(fixtures)
                    if row > 0:
                        fixtures[0:row,...], fail_flag = shuffle_block(fixtures[0:row,...])
                    fixtures[row:,:,:(col+1)], fail_flag = refit_matches(fixtures[row:,:,:(col+1)])
                    if fail_flag:
                        if cur_rounds < size(best_fixtures, 0): best_fixtures = fixtures
                        fixtures = make_all_fixtures(team_groupings, pitches, reverse=reverse)
                        new_size = size(fixtures, 0)
                        if new_size < cur_rounds and new_size < round_target + 5:
                            print "reduced to " + str(new_size) + " rounds (" + str(round_target) + ")"
                        cur_rounds = new_size
                test_row = cur_rounds - 1
                test_pitch = 0
                fit_row = 0
                times_permuted += 1
                continue
            else:
                test_row = cur_rounds - 1
                fit_row += 1
                continue
        if (team_plays(fixtures, team_one, fit_row) or team_plays(fixtures, team_two, fit_row)):
            test_pitch += 1
            if test_pitch >= pitches:
                test_pitch = 0
                test_row -= 1
            continue
        fit_pitch = first_gap(fixtures, fit_row)
        if fit_pitch == -1:
            return fixtures
        move_games(fixtures, fit_row, fit_pitch, test_row, test_pitch)
        rows_permuted = False
        fixtures = remove_blank_rounds(fixtures)
        fixtures = sort_rows(fixtures)
        fixtures = align_left(fixtures)
        new_size = size(fixtures, 0)
        if new_size < cur_rounds and new_size < round_target + 5:
            print "reduced to " + str(new_size) + " rounds"
        cur_rounds = new_size
        test_row = cur_rounds - 1
        fit_row = 0
    print "generated fixture list with " + str(cur_rounds) + " rounds"
    return fixtures

def check_integrity(fixtures):
    for rows in fixtures:
        if len(set(rows.flatten()).difference([0])) != len(rows.nonzero()[0]):
            print "Problem here:"
            print rows

def find_subarray(fixtures):
    row = min((fixtures == 0).nonzero()[0])
    col = max(fixtures[row,...].nonzero()[1])
    return row, col

def first_gap(fixtures, game_round):
    if (fixtures[game_round,...] != 0).all():
        return -1
    else:
        return  min(nonzero(fixtures[game_round,...] == 0)[1])

def team_plays(fixtures, team, game_round):
    return any(fixtures[game_round,...] == team)

def optimise_rounds(fixtures, cycles):
    num_rounds = size(fixtures, 0)
    count = 0
    best_score = fixtures_rest_score(fixtures)
    while count < cycles:
        row_one = int(random.random() * num_rounds)
        row_two = int(random.random() * num_rounds)
        new_fixtures = switch_rows(fixtures.copy(), row_one, row_two)
        if fixtures_rest_score(new_fixtures) < best_score:
            fixtures = new_fixtures
        count += 1
    return fixtures

def switch_rows(fixtures, row_one, row_two):
    temp = fixtures[row_one,...].copy()
    fixtures[row_one,...] = fixtures[row_two,...]
    fixtures[row_two,...] = temp
    return fixtures

def fixtures_rest_score(fixtures):
    tot_score = 0
    for t in range(1, fixtures.max()+1):
        tot_score += team_rest_score(fixtures, t)
    return tot_score

def team_rest_score(fixtures, team):
    play_rounds = (fixtures==team).sum(2).sum(1)
    base = 0
    score = 0
    for r in range(0,size(play_rounds)):
        if play_rounds[r]:
            base += 1
        else:
            score += 2**base
            base = 0
    score += 2**base
    return score

def optimise_pitches(fixtures, cycles):
    num_pitches = size(fixtures, 2)
    num_rounds = size(fixtures, 0)
    count = 0
    best_score = fixtures_pitch_score(fixtures)
    while count < cycles:
        row_to_shuffle = int(random.random() * num_rounds)
        new_fixtures = row_shuffle(fixtures.copy(), row_to_shuffle)
        if fixtures_pitch_score(new_fixtures) > best_score:
            fixtures = new_fixtures
        count += 1
    return fixtures

def row_shuffle(fixtures, row_to_shuffle):
    pitch_order = arange(size(fixtures, 2))
    random.shuffle(pitch_order)
    shuffle_row = fixtures[row_to_shuffle,...]
    shuffle_row = shuffle_row[:,pitch_order]
    fixtures[row_to_shuffle,...] = shuffle_row
    return fixtures

def fixtures_pitch_score(fixtures):
    tot_score = 0
    for t in range(1, fixtures.max()+1):
        tot_score += team_pitch_score(fixtures, t)
    return tot_score   

def team_pitch_score(fixtures, team):
    play_pitches = (fixtures==team).sum(0).sum(0)
    return (2**play_pitches).sum()

def move_games(fixtures, r_1, p_1, r_2, p_2):
    fixtures[r_1, 0, p_1], fixtures[r_2, 0, p_2] = fixtures[r_2, 0, p_2], fixtures[r_1, 0, p_1]
    fixtures[r_1, 1, p_1], fixtures[r_2, 1, p_2] = fixtures[r_2, 1, p_2], fixtures[r_1, 1, p_1]

def sort_rows(fixtures):
    reverse_rows = nonzero((fixtures == 0).sum(2).sum(1))
    blankgames = (fixtures == 0).sum(2).sum(1)
    fixtures = fixtures[blankgames.argsort(),...]
    return fixtures

def permute_rows(fixtures):
    rows_with_spaces = nonzero((fixtures == 0).sum(2).sum(1))[0]
    index_list = arange(size(fixtures,0))
    index_list[rows_with_spaces] = index_list[rows_with_spaces[::-1]]
    fixtures = fixtures[index_list,...]
    return fixtures

def refit_matches(fixtures):
    pitches = size(fixtures, 2)
    rows_with_spaces = nonzero((fixtures == 0).sum(2).sum(1))[0]
    rows_without_spaces = nonzero((fixtures[:,0,:] != 0).sum(1) == pitches)[0]
    candidates = []
    for r in rows_with_spaces:
        for p in range(pitches):
            if fixtures[r, 0, p] != 0:
                candidates.append([r, p])
    random.shuffle(candidates)
    availables = []
    for r in rows_without_spaces:
        for p in range(pitches):
            availables.append([r, p])
    random.shuffle(availables)
    total_changes = 0
    while candidates:
        test_game_one = candidates.pop()
        for test_game_two in availables:
            if games_swappable(fixtures, test_game_one, test_game_two):
                move_games(fixtures, test_game_one[0], test_game_one[1], test_game_two[0], test_game_two[1])
                total_changes += 1
                if random.random() < 0.05:
                    return fixtures, False
                break
    if total_changes == 0:
        return fixtures, True
    return fixtures, False

def shuffle_block(fixtures):
    pitches = size(fixtures, 2)
    rounds = size(fixtures, 0)
    split_round = int((rounds - 1) * random.random())
    rows_with_spaces = arange(0, split_round + 1)
    rows_without_spaces = arange(split_round + 1, rounds)
    candidates = []
    for r in rows_with_spaces:
        for p in range(pitches):
            if fixtures[r, 0, p] != 0:
                candidates.append([r, p])
    random.shuffle(candidates)
    availables = []
    for r in rows_without_spaces:
        for p in range(pitches):
            availables.append([r, p])
    random.shuffle(availables)
    total_changes = 0
    while candidates:
        test_game_one = candidates.pop()
        for test_game_two in availables:
            if games_swappable(fixtures, test_game_one, test_game_two):
                move_games(fixtures, test_game_one[0], test_game_one[1], test_game_two[0], test_game_two[1])
                total_changes += 1
                if random.random() < 0.05:
                    return fixtures, False
                break
    if total_changes == 0:
        return fixtures, True
    return fixtures, False

def games_swappable(fixtures, game_one, game_two):
    swap_one = fixtures[game_one[0],0,game_one[1]] not in fixtures[game_two[0],...].flatten() or fixtures[game_one[0],0,game_one[1]] in fixtures[game_two[0],:,game_two[1]].flatten()
    swap_two = fixtures[game_one[0],1,game_one[1]] not in fixtures[game_two[0],...].flatten() or fixtures[game_one[0],1,game_one[1]] in fixtures[game_two[0],:,game_two[1]].flatten()
    swap_three = fixtures[game_two[0],0,game_two[1]] not in fixtures[game_one[0],...].flatten() or fixtures[game_two[0],0,game_two[1]] in fixtures[game_one[0],:,game_one[1]].flatten()
    swap_four = fixtures[game_two[0],1,game_two[1]] not in fixtures[game_one[0],...].flatten() or fixtures[game_two[0],1,game_two[1]] in fixtures[game_one[0],:,game_one[1]].flatten()
    return swap_one and swap_two and swap_three and swap_four

def remove_blank_rounds(fixtures):
    fixtures = fixtures[logical_not(fixtures.sum(2).sum(1)==0),:,:]
    return fixtures

def minimum_rounds(groupings, pitches):
    gametotal = 0
    for t in groupings:
        gametotal += t * (t-1) / 2
    return -(-gametotal / pitches)

def calc_groupings(teams, groups):
    team_groupings = [teams / groups] * groups
    for a in range(teams % groups):
        team_groupings[a]+=1
    return team_groupings

def align_left(fixtures):
    rows_with_spaces = nonzero((fixtures == 0).sum(2).sum(1))[0]
    for r in rows_with_spaces:
        fixtures[r, ...] = shift_spaces(fixtures[r, ...])
    return fixtures

def shift_spaces(fixture_row):
    fixture_row = fixture_row[:,argsort(fixture_row)[0][::-1]]
    return fixture_row

def get_referees(fixtures):
    num_teams = fixtures.max()
    num_rounds = size(fixtures, 0)
    num_pitches = size(fixtures, 2)
    ref_games = zeros(num_teams, dtype=int8)
    next_ref = [0] * num_teams
    ref_matrix = zeros([num_rounds, 2, num_pitches], dtype=int8)
    for r in range(num_rounds):
        rest_teams = resting_teams(fixtures, r)
        if not rest_teams:
            print "No available referees in round " + str(r+1) + "!"
            return []
        for p in range(num_pitches):
            if fixtures[r, 0, p] != 0:
                next_team_up = least_times_reffed(ref_games, rest_teams)
                ref_matrix[r, 0, p] = next_team_up + 1
                ref_games[next_team_up] += 1
                ref_matrix[r, 1, p] = next_ref[next_team_up] + 1
                next_ref[next_team_up] = (next_ref[next_team_up] + 1) % 6
    return ref_matrix
        
def least_times_reffed(ref_games, rest_teams):
    return list(rest_teams)[argmin(ref_games[list(rest_teams)])]

def resting_teams(fixtures, round_num):
    num_teams = fixtures.max()
    rest_teams = set(range(num_teams))
    for t in fixtures[round_num,...].flatten():
        rest_teams = rest_teams.difference([t-1])
    return rest_teams

def save_fixtures(filename, fixtures, referees = array([])):
    FILE = open(filename, "w")
    FILE.write("Round,Pitch,Team One,Team Two")
    if referees.any():
        FILE.write(",Ref Team,Ref Player")
    FILE.write("\n")
    for r in range(size(fixtures,0)):
        for p in range(size(fixtures,2)):
            FILE.write(str(r + 1))
            FILE.write(",")
            FILE.write(str(p + 1))
            FILE.write(",")
            FILE.write(str(fixtures[r, 0, p]))
            FILE.write(",")
            FILE.write(str(fixtures[r, 1, p]))
            if referees.any():
                FILE.write(",")
                FILE.write(str(referees[r, 0, p]))
                FILE.write(",")
                FILE.write(str(referees[r, 1, p]))  
            FILE.write("\n")
    FILE.close()

def fixture_stats(fixtures, break_time, total_time):
    num_rounds = size(fixtures, 0)
    num_teams = fixtures.max()
    game_length = int(((total_time - break_time) / num_rounds) - break_time)
    games_played = histogram(fixtures.flatten(), range(1, num_teams + 2))[0]
    max_time_played = (max(games_played) * game_length)/float(total_time)
    min_time_played = (min(games_played) * game_length)/float(total_time)
    print "Game length is " + str(game_length)
    print "Highest playing proportion is " + str(max_time_played)
    print "Lowest playing proportion is " + str(min_time_played)
    return game_length, max_time_played, min_time_played

def fixtures_html(fixtures):
    templateLoader = jinja2.FileSystemLoader( searchpath=getcwd() )
    templateEnv = jinja2.Environment( loader=templateLoader )
    template = templateEnv.get_template( "fixturetemplate.html" )
    tempfile = open('tempfile.html', 'w')
    tempfile.write(template.render(pitches=range(1, len(fixtures[0][0])+1), fixtures=[[(r[0][i], r[1][i]) for i in range(len(r[0]))] for r in fixtures]))
    tempfile.close()
    url = "file://" + getcwd() + "/tempfile.html"
    webbrowser.open(url, new=2)
