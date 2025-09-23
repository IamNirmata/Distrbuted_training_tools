#plan
nvidia dgx B200 allpair test plan
    1. Pull allpair code from public github
    2. Fetch node names from hostfile
    3. Python code generate permutations N =(n(n−1))/2
    4. Pass all pairs ( N) to the bash script
    5. Bash script runs all the pairs per bacth in parallel and wait -> then proceed to next batch
    6. Results get generated
    7. Using kubectl script , we map the node name from pod name ( I am trying to see if there is a way to get it from within the pod )
    8. Save results with 2 node names. Log
    9. Push the files to gcr-admin NFS storage


