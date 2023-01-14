import time
from datetime import datetime

import pandas as pd
from discord_webhook import DiscordWebhook, DiscordEmbed

COLORS = {
    "start": "ffffff",
    "error": "ff0000",
    "success": "00ff00",
    "blue": "03b2f8"
}


def get_webhook_url():
    """
    Returns webhook url or None if not found
    -------
    """
    try:
        import config.notification_settings as notification_settings
        return notification_settings.MY_HOOK_URL
    except ImportError:
        # ignore import error
        # print("Notification settings not found")
        return None


def get_df_type(df):
    first = pd.to_datetime(df.iloc[0]['date'])
    # TODO: handle dynamic dataset size
    second = pd.to_datetime(df.iloc[10]['date'])
    delta = second - first
    if delta.days == 1:
        return "1d"
    if delta.seconds >= 3600:
        return f"{int(delta.seconds / 3600)}h"
    if delta.seconds >= 60:
        return f"{int(delta.seconds / 60)}min"
    return f"{int(delta.seconds)}sec"


def get_time():
    now = datetime.now()
    return now.strftime("%d.%m.%Y %H:%M:%S")


def log_duration(duration, name=""):
    name = f"{name} " if name != "" else ""
    print(f"{get_time()}: {name}finished in {duration}")


def log_duration_from_start(start_time, name=""):
    duration_string = get_duration(time.time() - start_time)
    log_duration(duration_string, name)


def get_duration(duration):
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'


def get_embedding(title, description, color, strategy, df_type, run_cfg, device, timesteps):
    embed = DiscordEmbed(title=title, description=description, color=color)
    # embed.set_author(name='DRL Library', icon_url='https://avatars.githubusercontent.com/u/63058869')
    # embed.set_footer(text=f'Finished in {duration_string}')
    embed.set_timestamp()
    embed.add_embed_field(name='Strategy', value=strategy, inline=True)
    embed.add_embed_field(name='Data', value=df_type, inline=True)
    embed.add_embed_field(name='Config', value=run_cfg, inline=False)

    embed.add_embed_field(name='Device', value=device, inline=True)
    embed.add_embed_field(name='Timesteps', value=f"{timesteps:0_.0f}", inline=True)

    return embed


def log_start(settings, env_kwargs, df, send_discord=True):
    strategy_name = env_kwargs["model_name"]
    run_config = env_kwargs['run_name']
    df_type = get_df_type(df)
    device = settings['model_params']['device'] if 'device' in settings['model_params'] else "unknown"
    timesteps = settings["total_timesteps"]
    url = get_webhook_url()
    if not url or not send_discord:
        return

    webhook = DiscordWebhook(url=url)  # , content=message)
    embed = get_embedding("Training started", None, COLORS['start'],
                          strategy_name, df_type, run_config, device, timesteps)
    webhook.add_embed(embed)
    webhook.execute()


def log_finished(success, start_time, settings, env_kwargs, df, error=None, send_discord=True):
    duration_string = get_duration(time.time() - start_time)
    log_duration(duration_string)

    url = get_webhook_url()
    if not url or not send_discord:
        return

    strategy_name = env_kwargs["model_name"]
    run_config = env_kwargs['run_name']
    df_type = get_df_type(df)
    device = settings['model_params']['device'] if 'device' in settings['model_params'] else "unknown"
    timesteps = settings["total_timesteps"]

    if success:
        mycolor = COLORS['success']
        title = f"Training completed in {duration_string}"
        description = None
    else:
        mycolor = COLORS['error']
        title = f"Training failed after {duration_string}"
        description = f"{type(error).__name__}: {str(error)}" if error else "Unknown error"

    webhook = DiscordWebhook(url=url)  # , content='Webhook Message')
    embed = get_embedding(title, description, mycolor,
                          strategy_name, df_type, run_config, device, timesteps)
    webhook.add_embed(embed)
    webhook.execute()
