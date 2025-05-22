import atlantis


def li():
    """
    List tool details
    """

    atlantis.client_command("\\remote refresh_all")
    atlantis.client_command("\\tool info")
