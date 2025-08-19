import os
import time
# Setting for multiprocessing using pytorch
import torch.multiprocessing as mp
from environment.data_loader import save_to_csv


def multiprocess_writer(queue: mp.Queue, log_path: str, metrics_path: str, data_level: str, verbose: bool):
    """
    A general-purpose multiprocess writer that listens for messages
    tagged as 'csv', 'log', or 'time elapsed', and handles them accordingly.
    """
    while True:
        msg = queue.get()
        if msg == "__STOP__":
            break

        tag = msg.get("tag")

        if tag == "csv":
            station_data = msg["station_data"]
            agent_data = msg["agent_data"]
            save_to_csv(station_data, f'{metrics_path}/metrics_station_{data_level}.csv', True)
            save_to_csv(agent_data, f'{metrics_path}/metrics_agent_{data_level}.csv', True)

        elif tag == "log":
            text = msg["data"]
            if verbose:
                with open(log_path, 'a', encoding='utf-8') as file:
                    print(text, file=file)
                    file.flush()                   # flush Python I/O buffer
                    os.fsync(file.fileno())        # flush OS buffer to disk
    
            print(text, flush=True)
        
        elif tag == "sustainability_episode":
            sustainability_data = [{
                "kwh": msg["kwh"],
                "co2": msg["co2"],
                "episode": msg["episode"],
                "zone_index": msg["zone_index"],
                "aggregation_step": msg["aggregation_step"]
            }]

            save_to_csv(sustainability_data, f'{metrics_path}/metrics_sustainability_episode.csv', True)

        elif tag == "sustainability_aggregation":
            sustainability_data = [{
                "model_size_kb": msg["model_size_kb"],
                "aggregation_step": msg["aggregation_step"]
            }]

            save_to_csv(sustainability_data, f'{metrics_path}/metrics_sustainability_aggregation.csv', True)


def printer_queue(queue: mp.Queue):
    """
    Returns logging functions that place messages in the shared queue.

    Parameters:
        queue (mp.Queue): Queue shared with log_writer

    Returns:
        print_l (function): Enqueue a log message
        print_elapsed_time (function): Enqueue an elapsed time message
    """
    def print_l(to_print):
        queue.put({
            'tag':'log',
            'data': to_print
        })

    def print_elapsed_time(msg, start_t):
        et = time.time() - start_t
        h = f"{int(et // 3600):02}:{int((et % 3600) // 60):02}:{int(et % 60):02}"
        queue.put({
            'tag':'log',
            'data': f'{msg} - {h}'
        })

    return print_l, print_elapsed_time