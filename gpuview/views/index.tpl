<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="GPUView">
    <meta name="author" content="Hao Ren">
    <title>GPUView</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous"/>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css" 
        rel="stylesheet" type="text/css"/>
    <link href="https://cdn.datatables.net/1.10.16/css/dataTables.bootstrap4.min.css" rel="stylesheet"/>
</head>

<body class="fixed-nav sticky-footer bg-dark" id="page-top">
    <div class="content-wrapper">
        <div class="container-fluid" style="padding: 20px 20px 20px 20px">
            <div class="row">
                % for gpu_stat in gpu_stats:
                % for gpu in gpu_stat.get('gpus', []):
                <div class="col-xl-3 col-md-4 col-sm-6 mb-3">
                    <div class="card text-white {{ gpu.get('flag', '') }} o-hidden h-100">
                        <div class="card-body">
                            <div class="float-left">
                                <div class="card-body-icon">
                                    <i class="fa fa-server"></i> <b>{{ gpu_stat.get('hostname', '-') }}</b>
                                </div>
                                <div>[{{ gpu.get('index', '') }}] {{ gpu.get('name', '-') }}</div>
                            </div>
                        </div>
                        <div class="card-footer text-white clearfix small z-1">
                            <span class="float-left">
                                <span class="text-nowrap">
                                <i class="fa fa-thermometer-three-quarters" aria-hidden="true"></i>
                                Temp. {{ gpu.get('temperature.gpu', '-') }}&#8451;
                                </span> |
                                <span class="text-nowrap">
                                <i class="fa fa-microchip" aria-hidden="true"></i>
                                Mem. {{ gpu.get('memory', '-') }}% 
                                </span> |
                                <span class="text-nowrap">
                                <i class="fa fa-cogs" aria-hidden="true"></i>
                                Util. {{ gpu.get('utilization.gpu', '-') }}%
                                </span> |
                                <span class="text-nowrap">
                                <i class="fa fa-users" aria-hidden="true"></i>
                                {{ gpu.get('users', '-') }}
                                </span>
                            </span>
                        </div>
                    </div>
                </div>
                % end
                % end
            </div>
            <!-- GPU Stat Card-->
            <div class="card mb-3">
                <div class="card-header">
                    <i class="fa fa-table"></i> All Hosts and GPUs</div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-bordered" width="100%" cellspacing="0">
                            <thead>
                                <tr>
                                    <th scope="col">Host</th>
                                    <th scope="col">GPU ID / Name</th>
                                    <th scope="col">Temp.</th>
                                    <th scope="col">Util.</th>
                                    <th scope="col">Mem. Use / Cap</th>
                                    <th scope="col">Power Use / Cap</th>
                                    <th scope="col">Users / Prog</th>
                                </tr>
                            </thead>
                            <tbody>
                                % for gpu_stat in gpu_stats:
                                % for gpu in gpu_stat.get('gpus', []):
                                <tr class="small">
                                    <th scope="row">{{ gpu_stat.get('hostname', '-') }} </th>
                                    <td> [{{ gpu.get('index', '') }}] {{ gpu.get('name', '-') }} </td>
                                    <td> {{ gpu.get('temperature.gpu', '-') }}&#8451; </td>
                                    <td> {{ gpu.get('utilization.gpu', '-') }}% </td>
                                    <td> {{ gpu.get('memory', '-') }}% ({{ gpu.get('memory.used', '') }}MB / {{ gpu.get('memory.total', '-') }}MB) </td>
                                    <td> {{ gpu.get('power.draw', '-') }}W / {{ gpu.get('enforced.power.limit', '-') }}W </td>
                                    <td> {{ gpu.get('user_processes', '-') }} </td>
                                </tr>
                                % end
                                % end
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="card-footer small text-muted">{{ update_time }}</div>
            </div>
        </div>
        <script src="https://code.jquery.com/jquery-3.3.1.min.js" 
            integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8="
            crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" 
            integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
            crossorigin="anonymous"></script>
        <script src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.10.16/js/dataTables.bootstrap4.min.js"></script>
    </div>
</body>

</html>
