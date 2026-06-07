while true; do 
  echo "Starting algorithm at $(date)..."; 
  (ulimit -v 2142880 && timeout --foreground 10h ./build/main  -i assets/maze-32-32-2.scen -m assets/maze-32-32-2.map -N 20 -v 3 -t 20000 -O 2 -wdg -seed 1)
  echo "Process exited or was killed. Restarting in 5 seconds..."; 
  sleep 5; 
done
