337827265
321658379
*****
Comments:


    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:

    we spent a lot of time trying to find heuristics that work well
    initially we created a linear combination, multiplying by the score as the base.
    with bonuses for having the max tile in the corner,
    for having certain rows monotonically increase to this corner,
    for having more of the board free (either as a sum or a percentage)

    while we sometimes achieved nice results,
    the algorithm could not recover when forced to leave the corner,
    as suddenly the high penalties for corner and monotonicity
    prevented the moves necessary to restore a well balanced board.

    By applying a pattern as the base score, we changed the base multiplier to be a gradient
    where moving higher tiles to the corner always improves their value.
    suddenly the algorithm was able to recover from states even we would have given up on.
    was very impressive to see.
