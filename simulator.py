import polars as pl
import numpy as np
import random
import time
import warnings
import munkres
import sys
from collections import Counter, defaultdict
from convert_func import converter

warnings.filterwarnings("ignore",
                        category=pl.exceptions.MapWithoutReturnDtypeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


iters = 10000

# Parameters to tune
skill_sd = 2    # For distribution of team skill (1.2 is usual)
variation = 1   # For how variable teams' performance generally is
drop_rate = 0   # Expected number of rooms of non-participating teams
teams = 300     # How many teams (300 worlds, 160 euros)
num_rounds = 9

BIASES = []



def worlds_gen(num_teams, skill_sd):
    """
    Returns the base dataframe which a wudc is run on
    Generates random 8-char name (2e-7 collision prob for a 300-team comp)
    Only bit that isn't self-explanatory is the position names
    which count how many times a given team has been in that position
    """

    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    wudc = pl.DataFrame({"name": [''.join(random.choice(uppercase)
                            for a in range(8))
                            for b in range(num_teams)],
                        "skill": np.random.normal(0, skill_sd, num_teams),
                        "points": 0.0,
                        "og": 0,
                        "oo": 0,
                        "cg": 0,
                        "co": 0})

    return wudc


def get_bias(round_num):
    """
    Returns a list, which is the amount the motion favours a given position
    e.g., (1, 0.5, -0.5, -1) means teams that speak earlier do much better
    """
    # return [0, 0, 0, 0]
    bias = np.array([1])
    while abs(bias.sum()) > 0.01:
        bias = (
            np.random.normal(0, 0.34, 4) + np.array([0.05, 0.40, -0.35, -0.10])
        )
        # bias = np.random.uniform(-1.2, 1.2, 4)
    BIASES.append(list(bias))
    return list(bias)


def skipper(wudc, round_num, drop_rate):
    """
    This does a full round (one of nine) of worlds
    Sadly this takes a while (~2.5-3.0s) and it cannot be optimised
    This is because about 95% of computing time is used by the Munkres solver
        (80% of which is spent on rounds 4 and 8)
    And that is the official way that rounds are calculated
    Doing something else would almost certainly yield very biased results
    """
    # print(f"Round {round_num}")

    def get_cost_matrix():
        """
        Gives the matrix necessary for calculating the draw
        Each team has one row
        Each spot in the round's draw (e.g., OO in room 5) has one column
        The value in a given cell is the cost of puttin that team in that spot
        This depends on two things:
            1. The number of time they have already been in that position
            2. The bracket they are in
        """

        def get_cost_col(position):
            """
            Calculates the cost of putting a team into a given position
            for the next round. Details can be found in the Tabbycat docs
            https://tabbycat.readthedocs.io/en/stable/features
            /draw-generation-bp.html
            """
            exponent = 4
            updated = wudc.select(pl.col("og"),
                                pl.col("oo"),
                                pl.col("cg"),
                                pl.col("co")
                            ).with_columns(
                                (pl.col(position) + 1).alias(position)
                            ) / round_num
            updated = (updated
                        * pl.DataFrame(np.log2(updated),
                                        schema=pos)
                        ).fill_nan(0) * -1
            cost = ((2 - updated.sum_horizontal()) * round_num) ** exponent
            return cost

        def get_full_cost_matrix():
            """
            As the name says, makes the cost matrix, i.e., the 300x300 grid
            which has the cost of putting a team in each position.
            This does not yet account for brackets.
            """
            four_col = pl.DataFrame(dict(zip(pos,
                                            [get_cost_col(p) for p in pos])))
            copies = [four_col.clone().rename(lambda x: f"{i+1}_" + x)
                        for i in range(int(wudc.height / 4))]
            conjoined = pl.concat(copies, how="horizontal")
            return conjoined

        def add_disalloweds_wudc(cost_matrix):
            """
            This is what adds the brackets to the cost matrix
            """
            def do_pullups(points):
                """
                This is taken straight from the Tabbycat source code
                https://github.com/TabbycatDebate/tabbycat/blob/develop
                /tabbycat/draw/generator/bphungarian.py
                Function starts on line 111
                """
                counts = Counter(points)
                rooms = []
                allowed = set()
                nteams = 0
                level = None
                pullups_needed = 0
                for p in sorted(counts.keys(), reverse=True):
                    if pullups_needed < counts[p]:
                        if pullups_needed:
                            allowed.add(p)
                            counts[p] -= pullups_needed
                            nteams += pullups_needed
                        assert nteams % 4 == 0
                        rooms += [(level, allowed)] * (nteams // 4)
                        nteams = 0
                        allowed = set()
                        level = None

                    if counts[p] > 0:
                        allowed.add(p)
                        if level is None:
                            level = p
                    nteams += counts[p]
                    pullups_needed = (-nteams) % 4

                assert nteams % 4 == 0
                rooms += [(level, allowed)] * (nteams // 4)

                return rooms

            points = ()
            rooms = do_pullups(wudc.get_column("faux_points").to_list())
            low_scores = [min(rooms[i][1]) for i in range(len(rooms))]
            high_scores = [max(rooms[i][1]) for i in range(len(rooms))]
            penalties = wudc.select([(((pl.col("faux_points") < low_scores[i])
                                    + (pl.col("faux_points") > high_scores[i]))
                                    * sys.maxsize
                                    ).alias(f"{i}_{p}")
                                    for i in range(len(rooms))
                                    for p in pos])
            return cost_matrix + penalties

        return add_disalloweds_wudc(get_full_cost_matrix())

    pos = ["og", "oo", "cg", "co"]

    # Step 1: Generate the draw
    num_teams_missing = 4 * np.random.poisson(drop_rate)
    doing_next_round = pl.Series([0] * num_teams_missing +
                                [1] * (wudc.height - num_teams_missing))
            # 1a: Decide how many teams are missing
            # (currently this is always 0)
    wudc = wudc.sort(pl.col("points")
                        + np.random.normal(0, round_num / 2, wudc.height)
            ).with_columns(
                (((pl.col("points") + 1)
                    * doing_next_round) - 1).alias("faux_points")
            ).sort(pl.col("faux_points"), descending=True)
            # 1b: Have that many teams miss out (lower points -> higher prob)
    cost_matrix = get_cost_matrix()
    indices = munkres.Munkres().compute(cost_matrix.to_numpy())
            # 1c: Generate the draw according to convention
    destinations = pl.Series([index[1] for index in indices])
    room_num = np.floor(destinations / 4) + 1
    position = (destinations % 4).replace_strict(
        {0: "OG", 1: "OO", 2: "CG", 3: "CO"}
    )

    # Step 2: Apply draw, get results, add to the table
    po = (destinations % 4).to_dummies().rename(lambda x: pos[int(x[1])])
    wudc = wudc.with_columns(
        pl.Series(f"r{round_num}_room", room_num),
        pl.Series(f"r{round_num}_position", position),
        pl.Series(f"temp_pos", destinations)
            # 2a: Add the draw to the wudc dataframe
    ).with_columns(
        [(wudc.get_column(p) + po.get_column(p)).alias(p) for p in pos]
            # 2b: Update position count
    ).sort(
        pl.col(f"temp_pos")
    )
    wudc = wudc.with_columns(
        pl.Series(
            f"r{round_num}_performance",
            np.random.gumbel(
                (
                    wudc.get_column("skill")
                    + pl.Series(get_bias(round_num) * int(wudc.height / 4))
                ),
                variation
            )
        )
            # 2d: Get each team's peformance in the round
    ).sort(
        (
            pl.col(f"r{round_num}_room") * 10**8 * -1
            + pl.col(f"r{round_num}_performance")
        ),
        descending=True
            # 2e: Sort so it goes first to fourth within each room
    ).with_columns(
        pl.Series(
            f"r{round_num}_result",
            [3,2,1,0] * int((wudc.height-num_teams_missing)/4)
                + [0] * num_teams_missing)
            # 2f: Fill in results, with the missing teams getting 0
    ).with_columns(
        sum([pl.col(f"r{i}_result") for i in range(1, round_num+1)]
        ).alias("points")
            # 2g: Update total results
    )

    return wudc


def run_worlds(num_teams, skill_sd, drop_rate):
    """
    Runs a whole wudc
    Returns a dataframe which has information on how every team performed
    """
    wudc = worlds_gen(num_teams, skill_sd)
    for i in range(1, num_rounds + 1):
        wudc = skipper(wudc, i, drop_rate)
    return wudc


def transcribe(wudc, num):
    """
    This writes to file in the split format
    So it can be read in easily to do logistic regression on
    """
    lines = []
    for round_num in range(1, num_rounds + 1):
        sub_frame = wudc.select(
            pl.col("name"),
            pl.col(f"r{round_num}_room"),
            pl.col(f"r{round_num}_position"),
            pl.col(f"r{round_num}_result")
        )
        for room in sub_frame.select(
            pl.col(f"r{round_num}_room").unique()
        ).to_series():
            room_frame = sub_frame.filter(pl.col(f"r{round_num}_room") == room)
            pos_list = []
            to_grab = [0, 2, 3]
            for p in ["OG", "OO", "CG", "CO"]:
                row = room_frame.filter(pl.col(f"r{round_num}_position") == p)
                pos_list.append([(row.transpose()[i].item()) for i in to_grab])
            for i in range(4):
                for j in range(i + 1, 4):
                    lines.append(
                        [
                            pos_list[i][0],
                            pos_list[j][0],
                            pos_list[i][1],
                            pos_list[j][1],
                            str(4 - int(pos_list[i][2])),
                            str(4 - int(pos_list[j][2])),
                            str(round_num)
                        ]
                    )
    write_file = open(f"posterior_sim_splits/output_{num}.txt", "w")
    write_file.write("team_a team_b team_a_pos team_b_pos ")
    write_file.write("team_a_rank team_b_rank round_num\n")
    for line in lines:
        write_file.write(" ".join(line) + "\n")
    return


def write_bias_frame():
    """
    Records the bias of each round
    """
    write_df = pl.DataFrame(
        {
            "sim": [i for i in range(iters) for _ in range(9)],
            "round": [i for i in range(1, 10)] * iters,
            "og_act": [bias[0] for bias in BIASES],
            "oo_act": [bias[1] for bias in BIASES],
            "cg_act": [bias[2] for bias in BIASES],
            "co_act": [bias[3] for bias in BIASES]
        }
    )
    write_df.write_csv("posterior_bias_save.txt", separator="\t")


def full_run(teams, skill_sd, iters, drop_rate):
    """
    Does a large number of simulations
    """
    global_start = time.time()
    # tables = []
    print("Starting...")
    for i in range(iters):
        # Do the work
        start = time.time()
        wudc = run_worlds(teams, skill_sd, drop_rate)
        # converted = converter(wudc)
        # tables.append(converted)

        # Do some printing
        message_a = f"Iteration number: {i+1}; Elapsed time: "
        el = time.time() - global_start
        h, m, s = int(el // 3600), int((el % 3600) // 60), int((el % 60) // 1)
        message_b = f"{h:02d}:{m:02d}:{s:02d}"
        sys.stdout.write(f"\r" + message_a + message_b)
        sys.stdout.flush()
        transcribe(wudc, i)
        # wudc.write_csv(f"sim_tables/table_{i}.txt", separator="\t")

    # huge = pl.concat(tables)
    # huge.write_csv("output.txt", separator="\t")
    write_bias_frame()
    print(f"\nFinished\nTook {(time.time() - global_start):.2f}s")




full_run(teams, skill_sd, iters, drop_rate)
