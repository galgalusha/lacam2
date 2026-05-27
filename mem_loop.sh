while true; do 
  echo "Starting algorithm at $(date)..."; 
  (ulimit -v 4242880 && timeout --foreground 6h ./build/main -i assets/random-11-5.scen -m assets/random-11-5.map -N 20 -v 3 -t 12000 -O 2 -seed 29011981 -max_ll_depth 5 -wdg)
  echo "Process exited or was killed. Restarting in 5 seconds..."; 
  sleep 5; 
done
