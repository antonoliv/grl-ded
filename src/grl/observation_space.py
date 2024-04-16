obs_keys = [
    'gen_p',
    'gen_q',
    'gen_theta',
    'gen_v',

    'gen_p_before_curtail',

    'load_p',
    'load_q',
    'load_theta',
    'load_v',

    'line_status',
    'rho'
    ]

storage_keys = [
    'storage_charge',
    'storage_power',
    'storage_power_target',
    'storage_theta'
    ]

delta_time = [
    'delta_time',
]

time_keys = [
    'minute_of_hour',
    'hour_of_day',
    'day',
    'day_of_week',
    'month',
    'year',
]

timestep_overflow_keys = [
    "timestep_overflow"
]

maintenance_keys = [
    'duration_next_maintenance',
    'time_next_maintenance',

    'time_before_cooldown_line',
    'time_before_cooldown_sub',
]


def get_obs_keys(storage: bool, maintenance: bool, delta_time: bool, time: bool, timestep_overflow: bool):
    ret = list(obs_keys)
    if storage:
        ret.extend(storage_keys)

    if delta_time:
        ret.extend(time_keys)

    if time:
        ret.extend(time_keys)

    if maintenance:
        ret.extend(maintenance_key)

    if timestep_overflow:
        ret.extend(timestep_overflow_keys)

    return ret    
