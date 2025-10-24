"""
Grouped MatMul as described in Triton Docs
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

And really helpful explanation from Evintunador
https://github.com/evintunador/triton_docs_tutorials/blob/main/06_matmul/matmul.py

C = A@B -> [M x K] @ [K x N] = [M x N]
"""

import os
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import triton
import triton.language as tl
from .flags import DLPACK_DISABLE

####################################################
### PSEUDOCODE TO UNDERSTAND INDEXING FOR MATMUL ###
####################################################

def naive_matmul_psuedo(A, B):

    """
    Matmul is simple! We have two matricies:

    A: [6 x 3]
    B: [3 x 8]
    C: [6 x 8] <- Output

    And what we want is to do a dot product between 
    every row of A and every col of B, and store it 
    in our new matrix C of shape [M x N]

    [a11 a12 a13]      [b11 b12 b13 b14 b15 b16 b17 b18]     [c11 c12 c13 c14 c15 c16 c17 c18]    
    [a21 a22 a21]      [b21 b22 b23 b24 b25 b26 b27 b28]     [c21 c22 c23 c24 c25 c26 c27 c28]
    [a31 a32 a33]  @   [b31 b32 b33 b34 b35 b36 b37 b38] =   [c31 c32 c33 c34 c35 c36 c37 c38]
    [a41 a42 a43]                                            [c41 c42 c43 c44 c45 c46 c47 c48]
    [a51 a52 a53]                                            [c51 c52 c53 c54 c55 c56 c57 c58]
    [a61 a62 a63]                                            [c61 c62 c63 c64 c65 c66 c67 c68]

    so c11 = dot([a11, a12, a12], [b11, b21, b31])
       c21 = dot([a21, a22, a22], [b11, b21, b31])
       ...

    Now we have a for loop here, but technically, each m,n combination
    will be represented by a thread. We have a total of m * n outputs
    so we will have m * n threads.

    And we can figure out our actual (m,n) index from there

    """

    M, K = A.shape
    K, N = B.shape
    C = torch.zeros(M, N)

    def dot(v1, v2):
        accum = 0
        for p,q in zip(v1, v2):
            accum += p*q
        return accum

    num_pids = list(range(M*N)) #[0, 1, 2, ... m*n]
    for i in num_pids:
        ### Get our m,n index from our position in the threads ###
        ### Each row of our output matrix is N long, so we can get ###
        ### which row we are on with a simple division operation ###
        m = i // N

        ### And we can get which column we are on with a simple modulo operation ###
        n = i % N

        C[m,n] = dot(A[m, :], B[:, n])
            
    return C

def blocked_matmul_pseudo(A, B, BLOCK_SIZE_M, BLOCK_SIZE_N):

    """
    We typically do GPU compute in blocks. Instead of having every
    thread of the GPU process a single row of A and a single col of B
    like we did earlier, we will have it process a block of A and a block of B.

    For simplicity we assume that the number of rows in A are divisible by BLOCK_SIZE_M
    and the number of cols in B are divisible by BLOCK_SIZE_N
    
    
    Lets say we were doing the following matmul:

    A: [6 x 3]
    B: [3 x 8]
    C: [6 x 8] <- Output
    BLOCK_SIZE_M = 3
    BLOCK_SIZE_N = 4

    [a11 a12 a13]      [b11 b12 b13 b14 b15 b16 b17 b18]     [c11 c12 c13 c14 c15 c16 c17 c18]    
    [a21 a22 a21]      [b21 b22 b23 b24 b25 b26 b27 b28]     [c21 c22 c23 c24 c25 c26 c27 c28]
    [a31 a32 a33]  @   [b31 b32 b33 b34 b35 b36 b37 b38] =   [c31 c32 c33 c34 c35 c36 c37 c38]
    [a41 a42 a43]                                            [c41 c42 c43 c44 c45 c46 c47 c48]
    [a51 a52 a53]                                            [c51 c52 c53 c54 c55 c56 c57 c58]
    [a61 a62 a63]                                            [c61 c62 c63 c64 c65 c66 c67 c68]
                                           

    Before, each c was computed by a single thread. This is pretty inefficient, so instead lets compute
    blocks of c. We already stated that the BLOCK_SIZE_M is 3 and BLOCK_SIZE_N is 4, so lets do that!

    [c11 c12 c13 c14 | c15 c16 c17 c18]    
    [c21 c22 c23 c24 | c25 c26 c27 c28]
    [c31 c32 c33 c34 | c35 c36 c37 c38]
    -----------------------------------
    [c41 c42 c43 c44 | c45 c46 c47 c48]
    [c51 c52 c53 c54 | c55 c56 c57 c58]
    [c61 c62 c63 c64 | c65 c66 c67 c58]
    
    Now what we want is for every thread block to process each block of our output. In this case we will
    have 4 different blocks, which we can write as:


    [B1 B2]
    [B3 B4]

    where B1 = [c11 c12 c13 c14]    
               [c21 c22 c23 c24]
               [c31 c32 c33 c34]

          B2 = [c15 c16 c17 c18]    
               [c25 c26 c27 c28]
               [c35 c36 c37 c38]
        
          ...


    And we can see each block refers to a block of rows and cols in our A and  respectively, for example:

    B1 = [c11 c12 c13 c14]    [a11 a12 a13]    [b11 b12 b13 b14] 
         [c21 c22 c23 c24]  = [a21 a22 a23] @  [b21 b22 b23 b24] 
         [c31 c32 c33 c34]    [a31 a32 a33]    [b31 b32 b33 b34] 

    ...

    """
    
    M, K = A.shape
    K, N = B.shape
    C = torch.zeros(M, N)

    assert M % BLOCK_SIZE_M == 0, "Assuming M divisible by BLOCK_SIZE_M for simplicity"
    assert N % BLOCK_SIZE_N == 0, "Assuming N divisible by BLOCK_SIZE_N for simplicity"

    ### Get how many rows and cols we have at the Block level ###
    num_block_rows = M // BLOCK_SIZE_M
    num_block_cols = N // BLOCK_SIZE_N
    
    ### Now each pid refers to a block of our output, not just a single output
    num_pids = list(range(num_block_rows * num_block_cols))
    for i in num_pids:

        ### This gives us the top left index of every block 
        block_m = i // num_block_cols # What row are we on?
        block_n = i % num_block_cols # What col are we on?

        ### But we need to convert block_m, block_n to actual indexes in our data
        m = block_m * BLOCK_SIZE_M
        n = block_n * BLOCK_SIZE_N

        ### Grab the corresponding block of rows and cols from A and B
        block_A = A[m:m+BLOCK_SIZE_M, :]
        block_B = B[:, n:n+BLOCK_SIZE_N] 

        ### Now its basically a tiny matmul between block_A and block_B!
        C[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE_N] = naive_matmul_psuedo(block_A, block_B)

    return C

def grouped_blocked_matmul_psuedo(A, B, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE):

    """
    Blocking our matmul is a step in the right direction as it now enables proper
    usage of our GPUs. We can distribute each threadblock to process a specific block
    of our output matrix! But we have a different issue that is more subtle: Memory Access

    ### SUPER QUICK PRIMER ON GPUS ###
    Remember that the expensive part of our GPU compute typically comes down to loading 
    operations into our L2 Cache. 

    I really like this image to visualize it: https://supercomputingblog.com/cuda/cuda-memory-and-cache-architecture/

    Cache is the tiny bit of memory that sits between our cuda cores and the slower memory. 
    The amount of cache available is GPU dependent. For example, on my a6000 GPUs, I have:

    Memory: 48GB
    L1 Cache: 128kb (per SM) -> Each SM (streaming multiprocessor) is a group of cuda cores
    L2 Cache: 6mb

    So what are the differences here?
        Memory: This is the global (slow) memory on the GPU where it hold everything for the lifetime of the program
        Cache: A temporary scratchpad where we can copy data into from our slower Memory, but is temporary and managed automatically by CUDA
    
    The two different types of Cache are:

        L1 Cache is some memory for each SM to store the tiny bit of data it is processing. It is super fast
        but it local to that specific streaming multiprocessor

        L2 Cache is shared across ALL SMs on the GPU (slower than L1 but much faster than Memory). The benefit
        is multiple SMs can write to the same L2 cache at the same time, even through they are independent executions

    What we want to limit is repeated copying from Memory to Cache. These copies are expensive, and if we structure our problem 
    carefully we can potentially limit it! Ideally we can store something in Cache once, and then do lots of operations with it
    before we let it go (in triton we cant manually clear this, Tritons compiler does the heavy lifting for us!)
    
    TLDR: We want to maximize our Cache Hit Rate as much as possible! 

    ### BACK TO MATMUL ###

    In triton whenever we do tl.load() it essentially always is loading to cache. But, if it tires to load something, but it already has been loaded
    (potentially from a previous operation) then its a no-op! And we want to maximize this as much as possible, to reuse data already sitting in the 
    cache. 

    We are doing O = P @ Q

    P: [M x K]
    Q: [K x N]
    O: [M x N]

    And we saw ealier we want to organize it so every threadblock is processing a block of our output O
    Lets revisit this again, where each B is a Block of our output matrix. 

    [B11 B12 B13 B14]
    [B21 B22 B23 B24]
    [B31 B32 B33 B34]
    [B41 B42 B43 B44]

    So our BLOCK dimensions is 4 x 4 here, but each index represents BLOCK_SIZE_M x BLOCK_SIZE_N elements in our initial matrix

    Normally we process things in row major. We would in this case have 4 * 4 PIDs, and we would just go in order of 

    0 -> B11, 1 -> B12, 2 -> B13, 3 ->  B14, 4 -> B21, 5 -> B22, ...

    So our PIDs would simply be organized as:

    [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

    The issue is this, are we properly reusing data? We are processing things at the block level which is good, but 
    we may not be reusing stuff we copied to cache all that well. 

    Lets go Row Major to see what happens:

    ### BLOCK B11 (PID 0)###
    To compute B11 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[0:BLOCK_SIZE_M, :] (the first BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> Q[:, 0:BLOCK_SIZE_N] (the first BLOCK_SIZE_N COLUMNS and all the Rows)

        And we can do our tiny matmul on these two blocks of P and Q: block_p @ block_q -> [BLOCK_SIZE_M, K] @ [K, BLOCK_SIZE_N]

    ### BLOCK B12 (PID 1) ###
    To compute B12 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[0:BLOCK_SIZE_M, :] (the first BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> Q[:, BLOCK_SIZE_N:2*BLOCK_SIZE_N] (the second BLOCK_SIZE_N COLUMNS and all the Rows)

        WAIT! our Block of Rows from P is still P[0:BLOCK_SIZE_M, :], which is the same as before! So we get to use a Cached P
        but have to load a new Q. 


    ### BLOCK B13 (PID 2) ###
    To compute B12 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[0:BLOCK_SIZE_M, :] (the first BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> P[:, BLOCK_SIZE_N:2*BLOCK_SIZE_N] (the second BLOCK_SIZE_N COLUMNS and all the Rows)

        WAIT! our Block of Rows from P is still P[0:BLOCK_SIZE_M, :], which is the same as before! So we get to use a Cached P
        but have to load a new Q.

    ...

    What you will see here is we get to use our Cached P as we process PIDs 0,1,2,3 but load a new column BLOCK of Q for every PID. 
    
    ### PROBLEM NUMBER 1: CACHING GOES BOTH WAYS

    We only benifit from caching in one dimension. As we go across the rows, we get to reuse the cached P, but in the same way, 
    as we go across the column we could also reuse cached Qs.

    ### PROBLEM NUMBER 2: WE ONLY HAVE SO MUCH MEMORY

    At the end of all of this we wrote 4 blocks (the top row of blocks) to our output. To do this we had to load:

    BLOCK_SIZE_M * K values from P once and iteratively 4 * BLOCK_SIZE_N * K from Q. But Q is a K x N = K x 4*BLOCK_SIZE_N matrix. 
    Thats the entire Q matrix! And if Q is large enough, by the time we get to the end of our row of Blocks, the early rows will
    be released. (i.e. once we get to PID 3, due to cache limitations, the load of our first column block of Q could have been removed) 

    This means when we start on our second row at PID 4, we may have to RELOAD OUR FIRST COLUMN BLOCK OF Q!!

    ### PROBLEM NUMBER 3: MEMORY ACCESS PATTERNS

    Typically our matricies are stored in row major form. This means grabbing a row of data from the matrix is easy as 
    they will exist in consecutive memory positions. On the other hand, grabbing a column of data from the matrix is 
    much less efficient. As going down the column means getting every Nth element (assuming our matrix has N columns). 
    This means grabbing our column blocks from Q is much more expensive than grabbing Row Blocks from P. 


    ##### SOLUTION: GROUPED ORDERING ######

    What if instead of trying to compute our blocks like this (how we assign our PID to the different blocks):
    
    [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

    We did this instead:

    [0  2  4  6 ]
    [1  3  5  7 ]
    [8  10 12 14]
    [9  11 13 15]

    And to make it easier to see we can break it up like so (this is a group size of 2):

    [0  2  |  4  6 ]
    [1  3  |  5  7 ]
    ----------------
    [8  10 |  12 14]
    [9  11 |  13 15]

    Remember that each PID is responsible for its entire Block. All we have done is reorganize our PIDs so instead of
    traversing along a row of blocks, we instead group our blocks like so. 

    The execution order of our PIDs is based on the actual value of the PID. So PIDs 0,1,2 will occur before 13,14,15 for
    example. If we group our PIDs like this something really neat happens. Lets look at the top-left group


    ### BLOCK B11 (PID 0)###
    To compute B11 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[0:BLOCK_SIZE_M, :] (the first BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> Q[:, 0:BLOCK_SIZE_N] (the first BLOCK_SIZE_N COLUMNS and all the Rows)

        And we can do our tiny matmul on these two blocks of P and Q: block_p @ block_q -> [BLOCK_SIZE_M, K] @ [K, BLOCK_SIZE_N]

    ### BLOCK B21 (PID 1) ###
    To compute B21 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[BLOCK_SIZE_M:2*BLOCK_SIZE_M, :] (the second BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> Q[:, 0:BLOCK_SIZE_N] (the first BLOCK_SIZE_N COLUMNS and all the Rows)

        WAIT! our Block of Columns from Q is still  Q[:, 0:BLOCK_SIZE_N], which is the same as before! So we get to use a Cached Q
        but have to load a new P


    ### BLOCK B12 (PID 2) ###
    To compute B12 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[0:BLOCK_SIZE_M, :] (the first BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> Q[:, BLOCK_SIZE_N:2*BLOCK_SIZE_N] (the second BLOCK_SIZE_N COLUMNS and all the Rows)

        WAIT! our Block of Rows from P is still P[0:BLOCK_SIZE_M, :], which is the same as before (from PID 0)! So we get to use a Cached P
        but have to load a new Q.

    ### BLOCK B22 (PID 3) ###
    To compute B22 (the specfic block we are looking at), we need to copy in:

        Block of Rows from P -> P[BLOCK_SIZE_M:2*BLOCK_SIZE_M, :] (the second BLOCK_SIZE_M ROWS and all the Columns)
        Block of Columns from Q -> Q[:, BLOCK_SIZE_N:2*BLOCK_SIZE_N] (the second BLOCK_SIZE_N COLUMNS and all the Rows)
        
        WAIT! We have TWO REUSES NOW! We already loaded P[BLOCK_SIZE_M:2*BLOCK_SIZE_M, :] in PID 1 and we already loaded
        Q[:, BLOCK_SIZE_N:2*BLOCK_SIZE_N] in PID 2! We get to reuse both without any copying!
    

    ### SOLUTION TO PROBLEM 1: 
    We can see, now our caching can go both ways. We can reuse rows AND columns

    ### SOLUTION TO PROBLEM 2: 
    Because we have grouped things together that are close to each other, this maximizes the likelihood that when we go
    to do our operation, the data has already been copied in there, or we are copying it for reuse in an upcoming operation

    ### SOLUTION TO PROBLEM 3: 
    Notice that our grouping is:

    [0  2  |  4  6 ]
    [1  3  |  5  7 ]
    ----------------
    [8  10 |  12 14]
    [9  11 |  13 15]

    Not

    [0  1  |  4  5 ]
    [2  3  |  6  7 ]
    ----------------
    [8  9 |  12 13 ]
    [10 11 | 14 15 ]

    This is because like we discussed, accessing our column of data form Q is much more expensive than a row of data from P. 
    So we make sure we grab a column of Q once, complete all of our operations on it with easier to grab rows of P, before 
    we grab a new column

    At the end we still write 4 blocks to our output, but rather than a row of blocks, we are doing this grouping of blocks!

    ### CAVEAT ###
    We cannot explicitly tell CUDA to store specific things and remove specific things in Triton that easily. It is upto cuda
    to manage the cache for us. But by making consecutive threads have heavy overlap in the data that needs to be cached vs reloaded
    it makes it much more likely that the data that CUDA cached can be reused with an upcoming thread before it is cleared. 
    

    """

    M, K = A.shape
    K, N = B.shape
    C = torch.zeros(M, N)

    assert M % BLOCK_SIZE_M == 0, "Assuming M divisible by BLOCK_SIZE_M for simplicity"
    assert N % BLOCK_SIZE_N == 0, "Assuming N divisible by BLOCK_SIZE_N for simplicity"
    assert (BLOCK_SIZE_M % GROUP_SIZE == 0) and (BLOCK_SIZE_N % GROUP_SIZE == 0), "Assuming BLOCK_SIZE_M/N are divisible by our GROUP_SIZE"

    ### Get how many rows and cols we have at the Block level ###
    num_block_rows = M // BLOCK_SIZE_M
    num_block_cols = N // BLOCK_SIZE_N

    num_pids = list(range(num_block_rows * num_block_cols))

    ### To visualize our groups we can just store the PID index in a matrix to print
    pid_visual = torch.zeros(num_block_rows, num_block_cols)
    for i in num_pids:
        
        ### Get number of Programs in Group ###
        ### This is for all columns, as in our example ###
        ### col_id:    0  1     2  3   
        ### row_id: 0 [0  2  |  4  6 ]
        ### row_id: 1 [1  3  |  5  7 ]
        ###           ----------------
        ### row_id: 2 [8  10 | 12 14 ]
        ### row_id: 3 [9  11 | 13 15 ]

        ### to get to 8 (the start of the next group down the column)
        ### We need to get through 2 * 4 blocks. (group_size * num_block_cols)
        num_pid_in_group = GROUP_SIZE * num_block_cols

        ### Get Which Group We Are In (down the column) ###
        group_id = i // num_pid_in_group

        ### Now we can get the Row ID of the first program in the group 
        ### This is the top of this specific group, we need to index further
        ### soon to get the exact row we are processing inside the group
        first_pid_m = group_id * GROUP_SIZE

        ### As a sanity check our group_size may not perfectly divide our num blocks
        ### and the last group may be smaller. In this psuedocode we dont worry about 
        ### this as we assume divisible, but the triton code will have this so i put it here! 
        ### num_block_rows is how many total rows of blocks we have, and the first_pid_m
        ### is our starting row block. As we move down the rows, we may at the end not have
        ### GROUP_SIZE number of row blocks left (as its not perfectly divisble) so we just 
        ### update that here
        group_size_m = min(num_block_rows - first_pid_m, GROUP_SIZE)

        ### Now that we have our row we are working on we need the column. And as we saw
        ### we do column major order within groups. (this part is trickier)

        ### i % num_pid_in_group -> i can be anything from 0 to 15 in our setup and 
        ### num_pid_in_group is 8 for us. 
        ### if i == 0, i % 8 = 0
        ### if i == 1, i % 8 = 1
        ### if i == 2, i % 8 = 2
        ### ...
        ### if i == 7, i % 8 = 7
        ### if i == 8, i % 8 = 0
        ### Therefore, as our PID increases, we wil just go from 0 to 7 and then reset back to 0 again. 
        where_in_group = i % num_pid_in_group

        ### Now we need to get ourselves into the correct row, as its column major. This will be done 
        ### by doing where_in_group % group_size_m, where group_size_m = 2 for us 

        ### if i == 0, where_in_group = i % 8 = 0 % 2 = 0
        ### if i == 1, where_in_group = i % 8 = 1 % 2 = 1
        ### if i == 2, where_in_group = i % 8 = 2 % 2 = 0
        ### ...
        ### if i == 7, where_in_group = i % 8 = 7 % 2 = 1
        ### if i == 8, where_in_group = i % 8 = 0 % 2 = 0

        ### So we see toggle. i = 0 is placed in row 0 of our group.
        ###                   i = 1 is placed in row 1 of our group.
        ###                   i = 2 is placed in row 0 of our group again
        ###                   ...
        which_row = where_in_group % group_size_m

        ### And this is the row within the group, we just have to shift it over by where ever our starting group is
        pid_m = first_pid_m + which_row

        ### Now we get the column that this pid is supposed to compute. 
        ### we can just do first i % num_pid_in_group (same as before this is redundant)
        ### if i == 0, i % 8 = 0
        ### if i == 1, i % 8 = 1
        ### if i == 2, i % 8 = 2
        ### ...
        ### if i == 7, i % 8 = 7
        ### if i == 8, i % 8 = 0
        where_in_group = i % num_pid_in_group

        ### And just do division to get our actual row we are in!
        ### where_in_group // group_size_m
        ### if i == 0, where_in_group = i % 8 = 0 // 2 = 0
        ### if i == 1, where_in_group = i % 8 = 1 // 2 = 0
        ### if i == 2, where_in_group = i % 8 = 2 // 2 = 1
        ### if i == 3, where_in_group = i % 8 = 3 // 2 = 1
        ### if i == 4, where_in_group = i % 8 = 4 // 2 = 2
        ### if i == 5, where_in_group = i % 8 = 5 // 2 = 2
        ### ...
        pid_n = where_in_group // group_size_m
        ### so we can then construct:

        ### PID 0: (0,0)
        ### PID 1: (1,0)
        ### PID 2: (0,1)
        ### PID 3: (1,1)
        ### PID 4: (0,2)
        ### PID 5: (1,2)
        ### PID 6: (0,3)
        ### PID 7: (1,3)
        ### PID 8: (2,0)
        ### PID 9: (3,0)
        ### PID 10: (2,1)
        ### PID 11: (3,1)
        ### PID 12: (2,2)
        ### PID 13: (3,2)
        ### PID 14: (2,3)
        ### PID 15: (3,3)
        pid_visual[pid_m, pid_n] = i

        ### But we need to convert block_m, block_n to actual indexes in our data
        m = pid_m * BLOCK_SIZE_M
        n = pid_n * BLOCK_SIZE_N

        ### Grab the corresponding block of rows and cols from A and B
        block_A = A[m:m+BLOCK_SIZE_M, :]
        block_B = B[:, n:n+BLOCK_SIZE_N] 

        ### Same matmul op on our block of data ###
        C[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE_N] = naive_matmul_psuedo(block_A, block_B)

    return C, pid_visual

######################################################
### IDENTICAL TRITON CODE THAT DOES THE SAME THING ###
######################################################

@triton.jit
def naive_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn
):
    # Program ID — each "program" computes one element of C
    pid = tl.program_id(axis=0)  # 0..M*N - 1

    # Compute (m, n) coordinates of this output element
    m = pid // N
    n = pid % N

    # If we're out of bounds, exit
    if (m >= M) or (n >= N):
        return

    # Compute the dot product between A[m, :] and B[:, n]
    acc = 0.0
    for k in range(0, K):
        a_val = tl.load(A_ptr + m * stride_am + k * stride_ak)
        b_val = tl.load(B_ptr + k * stride_bk + n * stride_bn)
        acc += a_val * b_val

    # Write result to output
    tl.store(C_ptr + m * stride_cm + n * stride_cn, acc)

def naive_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    grid = (M * N,)

    naive_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )
    return C

def get_blocked_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
    ]
@triton.autotune(configs = get_blocked_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def blocked_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    
    pid = tl.program_id(axis=0)

    ### Get how many columns of blocks we have ###
    num_block_cols = tl.cdiv(N, BLOCK_SIZE_N)

    ### Recover our actual block index ###
    block_m = pid // num_block_cols # What row are we on?
    block_n = pid % num_block_cols # What col are we on?
    
    ### Compute Offsets to grab full block of data ###
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    ### Initializer Accumulator - use float32 for accumulation ###
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    ### Loop over K in chunks of BLOCK_SIZE_K ###
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        ### Load Blocks of A and B with proper masking ###
        # Mask for A: check both M and K dimensions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Mask for B: check both K and N dimensions
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        ### Accumulate Result ###
        ### investigate later: There is a discrepancy between pytorch and triton in float32
        ### if we dont include allow_fp32=False we wil have a higher error, but also, 
        ### we then hurt performance on float32. Leaving as is for now! 

        ### This is a well documented issue
        ### https://github.com/triton-lang/triton/issues/4574
        ### https://github.com/triton-lang/triton/issues/5204
        ### https://github.com/triton-lang/triton/issues/2843

        accumulator += tl.dot(a, b) #allow_tf32=False
        
        ### Advance our Offsets to the next chunk of K ###
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    ### Cast accumulator to correct dtype ###
    c = accumulator.to(A_ptr.dtype.element_ty)

    ### Identify which block in our output C we will save this in ###
    c_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    
    ### Identify any invalid positions we dont want to save in ###
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    ### Save it! ###
    tl.store(C_ptr + c_offsets, c, mask=c_mask)

def blocked_matmul(a, b):
    
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    # Initialize output tensor with same dtype as input
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    # Launch kernel
    blocked_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]

@triton.autotune(configs = get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def grouped_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE_FLAG: tl.constexpr # 0 for float32, 1 for float16
):
    
    ### Grouping logic as described above! ###
    pid = tl.program_id(axis=0)

    ### Cast our Pointers to the Correct DTYPE ###
    if DTYPE_FLAG == 0:  # float32
        A_ptr = tl.cast(A_ptr, tl.pointer_type(tl.float32))
        B_ptr = tl.cast(B_ptr, tl.pointer_type(tl.float32))
        C_ptr = tl.cast(C_ptr, tl.pointer_type(tl.float32))
    elif DTYPE_FLAG == 1:  # float16
        A_ptr = tl.cast(A_ptr, tl.pointer_type(tl.float16))
        B_ptr = tl.cast(B_ptr, tl.pointer_type(tl.float16))
        C_ptr = tl.cast(C_ptr, tl.pointer_type(tl.float16))

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    ### EVERYTHING ELSE IS THE SAME AS BEFORE! ###
    ### Compute Offsets to grab full block of data ###
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    ### Initializer Accumulator - use float32 for accumulation ###
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    ### Loop over K in chunks of BLOCK_SIZE_K ###
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        ### Load Blocks of A and B with proper masking ###
        # Mask for A: check both M and K dimensions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Mask for B: check both K and N dimensions
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        ### Accumulate Result ###
        ### investigate later: There is a discrepancy between pytorch and triton in float32
        ### if we dont include allow_fp32=False we wil have a higher error, but also, 
        ### we then hurt performance on float32. Leaving as is for now! 

        ### This is a well documented issue
        ### https://github.com/triton-lang/triton/issues/4574
        ### https://github.com/triton-lang/triton/issues/5204
        ### https://github.com/triton-lang/triton/issues/2843
        accumulator += tl.dot(a, b)
        
        ### Advance our Offsets to the next chunk of K ###
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    ### Cast accumulator to correct dtype ###
    c = accumulator.to(A_ptr.dtype.element_ty)

    ### Identify which block in our output C we will save this in ###
    c_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    
    ### Identify any invalid positions we dont want to save in ###
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    ### Save it! ###
    tl.store(C_ptr + c_offsets, c, mask=c_mask)

def torch_fused_grouped_matmul(a, b):
    
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    # Initialize output tensor with same dtype as input
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    # Launch kernel
    grouped_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        DTYPE_FLAG=0 if a.dtype == torch.float32 else 1
    )
    return c

def fused_grouped_matmul(a, b, use_dlpack=True):

    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    assert (a.dtype == cp.float32) or (a.dtype == cp.float16), "Only support float32 or float16 ops"
    assert a.dtype == b.dtype, f"a and b have different precision types {a.dtype} and {b.dtype}"
    
    if not DLPACK_DISABLE and use_dlpack:
        a = from_dlpack(a)
        b = from_dlpack(b)

        # Initialize output tensor with same dtype as input
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        # Define grid
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        
        # Launch kernel
        grouped_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            DTYPE_FLAG=0 if a.dtype == torch.float32 else 1
        )

        return cp.from_dlpack(c)

    else:
        # Initialize output tensor with same dtype as input
        with cp.cuda.Device(a.device.id):
            c = cp.empty(
                (M, N), dtype=a.dtype
            )

        # Define grid
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        
        # Launch kernel
        grouped_matmul_kernel[grid](
            a.data.ptr, 
            b.data.ptr, 
            c.data.ptr,
            M, N, K,
            a.strides[0]//a.itemsize, 
            a.strides[1]//a.itemsize,
            b.strides[0]//b.itemsize, 
            b.strides[1]//b.itemsize,
            c.strides[0]//c.itemsize, 
            c.strides[1]//c.itemsize,
            DTYPE_FLAG=0 if a.dtype == cp.float32 else 1
        )
        
        return c