import atlantis


def ls():
    """
    List tools and parameters
    """

    atlantis.client_command("\\remote refresh_all")
    atlantis.client_command("\\tool list")
