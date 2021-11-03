import json
import logging
import os
from urllib.request import urlopen

from bottle import TEMPLATE_PATH

abs_path = os.path.dirname(os.path.realpath(__file__))
abs_views_path = os.path.join(abs_path, 'views')
host_db_path = os.path.join(abs_path, 'hosts.db')
TEMPLATE_PATH.insert(0, abs_views_path)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y/%m/%d %H:%M:%S %p'
logging.basicConfig(filename='/tmp/gpuview_error.log', level=logging.ERROR, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def local_gpu_stat():
    """
    Returns gpu stat for this host.
        # Set color flag based on gpu temperature
        # bg-warning, bg-danger, bg-success, bg-primary

    Returns:
        dict: gpu stat
    """

    try:
        from gpustat import GPUStatCollection
        stat = GPUStatCollection.new_query().jsonify()
        delete_list = []
        for gpu_id, gpu in enumerate(stat['gpus']):
            if type(gpu['processes']) is str:
                delete_list.append(gpu_id)
                continue
            gpu['memory'] = round(float(gpu['memory.used']) / float(gpu['memory.total']) * 100)
            gpu['users'] = len(set([p['username'] for p in gpu['processes']]))
            user_process = ['%s(%s,%sM)' % (p['username'], p['command'], p['gpu_memory_usage'])
                            for p in gpu['processes']]
            gpu['user_processes'] = ' '.join(user_process)
            gpu['flag'] = 'bg-primary'
            if gpu['temperature.gpu'] > 75:
                gpu['flag'] = 'bg-danger'
            elif gpu['temperature.gpu'] > 50:
                gpu['flag'] = 'bg-warning'
            elif gpu['temperature.gpu'] > 25:
                gpu['flag'] = 'bg-success'
            gpu.pop('uuid')

        if delete_list:
            for gpu_id in delete_list:
                stat['gpus'].pop(gpu_id)
        stat.pop('query_time')
        return stat
    except Exception as e:
        return {'error': '%s!' % getattr(e, 'message', str(e))}


def all_gpu_stats():
    """
    Aggregates the gpu stats of all registered hosts and this host.

    Returns:
        list: gpu stats of hosts
    """

    gpu_stats, local_stat = [], local_gpu_stat()
    if 'gpus' in local_stat:
        gpu_stats.append(local_stat)

    hosts = load_hosts()
    for url in hosts:
        try:
            raw_resp = urlopen('https://' + url + '/gpustat')
            gpu_stat = json.loads(raw_resp.read())
            raw_resp.close()
            if not gpu_stat or 'gpus' not in gpu_stat:
                continue
            if hosts[url] != url:
                gpu_stat['hostname'] = hosts[url]
            gpu_stats.append(gpu_stat)
        except Exception as e:
            logging.error('Error: %s getting gpu stat from %s' % (getattr(e, 'message', str(e)), url))

    try:
        sorted_gpu_stats = sorted(gpu_stats, key=lambda g: g['hostname'])
        if sorted_gpu_stats is not None:
            return sorted_gpu_stats
    except Exception as e:
        logging.error('Error: %s' % getattr(e, 'message', str(e)))
    return gpu_stats


def load_hosts():
    """
    Loads the list of registered gpu host from file.

    Returns:
        dict: {url: name, ... }
    """

    hosts = {}
    if not os.path.exists(host_db_path):
        return hosts

    for line in open(host_db_path, 'r'):
        name, url = line.strip().split('\t')
        hosts[url] = name
    return hosts


def save_hosts(hosts):
    with open(host_db_path, 'w') as f:
        for url in hosts:
            f.write('%s\t%s\n' % (hosts[url], url))


def add_host(url, name=None):
    if name is None:
        name = url
    hosts = load_hosts()
    hosts[url] = name
    save_hosts(hosts)
    print('Successfully added host!')


def remove_host(url):
    hosts = load_hosts()
    if hosts.pop(url, None):
        save_hosts(hosts)
        print('Removed host: %s!' % url)
    else:
        print("Couldn't find host: %s!" % url)


def print_hosts():
    hosts = load_hosts()
    if len(hosts):
        hosts = sorted(hosts.items(), key=lambda g: g[1])
        print('#   Name\tURL')
        for idx, host in enumerate(hosts):
            print('%02d. %s\t%s' % (idx + 1, host[1], host[0]))
    else:
        print("There are no registered hosts! Use `gpuview add` first.")
