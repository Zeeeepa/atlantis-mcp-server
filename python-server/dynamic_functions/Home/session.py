import atlantis
import logging
import os

logger = logging.getLogger("mcp_server")

@public
@session
async def session():
    """
    Main session function
    """

    # get user id
    user_id = atlantis.get_caller()
    logger.info(f"Session started for user: {user_id}")

    # set background
    image_path = os.path.join(os.path.dirname(__file__), "builder.jpg")
    await atlantis.set_background(image_path)


    #await atlantis.client_command("\\chat set " + user_id + "*kitty")
    await atlantis.client_command("\\chat set " + user_id + "*kitty")

    # send kitty face image
    kitty_path = os.path.join(os.path.dirname(__file__), "kitty_face_compressed.jpg")
    await atlantis.client_image(kitty_path)

    await atlantis.client_log(f"Kitty is at the front desk! Hi, {user_id}!")




