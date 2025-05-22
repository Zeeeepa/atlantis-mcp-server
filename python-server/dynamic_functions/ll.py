import atlantis


def ll():
    """
    List tools by last modified date
    """

    atlantis.client_command("\\remote refresh_all")
    atlantis.client_command("\\tool date")
