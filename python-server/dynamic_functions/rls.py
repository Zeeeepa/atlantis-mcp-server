import atlantis


def rls():
    """
    List remotes
    """

    atlantis.client_command("\\remote refresh_all")
    atlantis.client_command("\\remote list")

