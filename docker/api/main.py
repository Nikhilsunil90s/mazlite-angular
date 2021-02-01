from utils.deps import *

from dotenv import load_dotenv
import os

load_dotenv()
# App builder
app = FastAPI()

# list of allowed origins
origins = [
    '*'
]

# add origins to app's middleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

duration = 0
sm=None
stats = {}
hist = {}
review_hist, review_stats = {}, {}
savingImages = 0
imgv = []
curr_meas = curr_proj = ''
has_cam = True
dataRoot = '/media/mazlite/data'
video_folder = 'Videos'
video_path = ''
_proj, _meas = '', ''
img_array, img_id = [], []
_ID_FILTER = 0
state = {'capture_start': False, 'review_start': False,
         'failed_op': False, 'failed_badly': False}
FMT = '%Y-%m-%d-%H-%M-%S'
# setup logger
logger = logutils.make_log(log_dir='./logs')

# mongo collections
mongoClient = None
mazliteDB = None
projectsCollection = None
userPreferenceCollection = None
measurementsCollection = None
iniCollection = None
metadataCollection = None
machineCollection = None

# curr_path
dir_path = os.path.dirname(os.path.realpath(__file__))

# temp
_frame = np.zeros((2076, 3088), dtype=np.uint8)

# structs
stream_settings = {'mark': False, 'boost': False, 'aoi': {'height': 2076, 'width': 3088, 'top': 0, 'left': 0},
                   'downsample': {'ratio': 4, 'height': 2076, 'width': 3088}, 'fps': 55, 'broadcastFPS': 15}

capture_settings = {'profile': 'default',
                    'num_particles_to_detect': 0, 'detect_particles': False, 'save_images': False, 'numberImages': 50}

operation_settings = {'measurement_mode': 'particle', 'capture_mode': 'Sequential',
                      'scan_mode': 'multi', 'measurement_type': 'full', 'trigger': 'Off'}

video_stream_info_settings = {'fps': '-1', 'duration': '-1', 'max_fps': '-1'}

cap_ops = {0: 'available', 1: 'processing review'}
review_ops = {0: 'available', 1: 'processing review', 2: 'upload'}

lights = {'rev': {'color': 'green', 'mode': 'solid', 'op': 'None'},
          'cap': {'color': 'green', 'mode': 'solid', 'op': 'None'},
          'proc': {'color': 'yellow', 'mode': 'blink', 'op': 'background processing'}}
help_settings = {'sw': '1.0', 'dw': '1.0', 'warranty': '', 'health': '',
                 'update': '', 'maintainance': '', 'optical': ''}

ins = {}


def line(e):
    print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))


def _get_status_message(n):
    """
    get a status message to send to the UI. 
    This is for now a test function but can be upgraded if needed
    If n < 0 or n > 7 this will return a solid green state with a test message
    """

    if n == 0:
        clr = 'Blue'
        fs = 'Solid'
        msg = 'The device is available for the review process.'
    elif n == 1:
        clr = 'Blue'
        fs = 'Blinking Slow'
        msg = 'The device is performing a review process.'
    elif n == 2:
        clr = 'Blue'
        fs = 'Blinking Fast'
        msg = 'The device is uploading data.'
    elif n == 3:
        clr = 'Green'
        fs = 'Solid'
        msg = 'The device is available for measurement.'
    elif n == 4:
        clr = 'Green'
        fs = 'Blinking Medium'
        msg = 'The device is capturing and processing.'
    elif n == 5:
        clr = 'Orange'
        fs = 'Blinking Slow'
        msg = 'The device is performing background processes.'
    elif n == 6:
        clr = 'Red'
        fs = 'Solid'
        msg = 'The device is in critical condition and needs immediate attention. Perform a restart.'
    elif n == 7:
        clr = 'Red'
        fs = 'Blinking Medium'
        msg = 'The device has failed one or more operations.'
    else:
        clr = 'Green'
        fs = 'Solid'
        msg = 'Test Message'

    return(clr, fs,  msg)


def prepareForStreaming(arg1, frame):
    # todo remove arg1 -> it's useless maybe not
    global _frame, savingImages, imgv, has_cam, stream_settings, sm
    # TODO: Change to static func#
    try:
        if has_cam:
            logging.debug("Preparing the image for streaming")
            sm = SystemMode.get_instance()
            sampling = stream_settings.get('downsample').get('ratio')
            stream_width = int(sm.get_current_thread().aoi.width / sampling)
            stream_height = int(sm.get_current_thread().aoi.height / sampling)
            resized_image = cv2.resize(frame, (stream_width, stream_height))
            _frame = cv2.imencode('.jpeg', resized_image, [
                                  cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tobytes()
            logging.info("Done with preparing the image for streaming")
            if savingImages > 0:
                savingImages -= 1
                image_name = IdsHelpers.GenerateImageName() + "_" + str(savingImages)  # + ".png"
                imgv.append(caps(name=image_name, img=frame, id=savingImages))
                #np.save(new_image_name, frame, allow_pickle=False)
                logging.info(
                    "Processed the saving event of the image {}".format(savingImages))
        else:
            resized_image = np.random.randint(
                255, size=(900, 800, 3), dtype=np.uint8)
            _frame = cv2.imencode('.jpeg', resized_image, [
                                  cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tobytes()
    except:
        pass


def _stream():
    global cnt, _frame
    while True:
        try:
            if _frame is not None:
                # get the next item
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + _frame + b'\r\n\r\n')
        # do something with element
        except Exception:
            logging.error('Frame cannot be updated for streaming')
            break
        except StopIteration:
            # if StopIteration is raised, break from loop
            break
# app generics


@app.on_event("startup")
async def startup_event():
    # needs to be global, maybe inherited later.
    global mongoClient, mazliteDB, metadataCollection, projectsCollection, measurementsCollection, userPreferenceCollection, iniCollection, machineCollection
    if not mongoClient:
        mongoClient = pymongo.MongoClient(
            "mongodb://{}:{}@{}:27017/".format(os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST")))
        if mongoClient is not None:
            logger.info('connected to database.')
            mazliteDB = mongoClient[os.getenv("DB_NAME")]
            projectsCollection = mazliteDB["projects"]
            measurementsCollection = mazliteDB["measurements"]
            userPreferenceCollection = mazliteDB["userPreference"]
            iniCollection = mazliteDB["configs"]
            metadataCollection = mazliteDB["metaData"]
            machineCollection = mazliteDB["machine"]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/404", include_in_schema=False)
async def err_404(request: Request):
    return templates.TemplateResponse("404.html", {"request": request})


@app.on_event("shutdown")
def shutdown_event():
    global mongoClient
    try:
        mongoClient.close()
    except Exception as e:
        logger.error(e)
    logger.info('app shutdown.')

# app routes


@app.get("/", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-login.html", {"request": request})


@app.get("/console", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-console.html", {"request": request})


@app.get("/project", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-project.html", {"request": request})


@app.get("/preset", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-project.html", {"request": request})


@app.get("/review", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-review.html", {"request": request})


@app.get("/measurement", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-measurement.html", {"request": request})


@app.get("/device", include_in_schema=False)
async def read_item(request: Request):
    return templates.TemplateResponse("index-project.html", {"request": request})


@app.post("/token", response_model=Token, include_in_schema=False)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(
        fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User, include_in_schema=False)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/", include_in_schema=False)
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "api_testing", "owner": current_user.username}]


@app.get("/stream")
def image_endpoint():
    return StreamingResponse(_stream(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/review/process')
def review(project_name: str, measurement_name: str):
    global _proj, _meas, savingImages, img_array, img_id, imgv, ph
    global state, has_cam, capture_settings, operation_settings, stream_settings
    global operation_settings, stats, hist, state, metadataCollection, video_path, review_hist, review_stats
    try:
        _state = state['review_start']
        _proj = project_name
        _meas = measurement_name
        if _state is False:
            state['review_start'] = True
            review_stats = ph.load_saved_stats(_proj, _meas)
            review_hist = ph.load_saved_histograms(_proj, _meas, 'both')
            ph.convert_to_png(_proj, _meas)
            video_path = ph.mp4_from_png(_proj, _meas, 15, (640, 480))
            state['review_start'] = False
            content = {}
            if review_stats != 1 and review_hist != 1:
                content.update(review_stats)
                content.update(review_hist)
            return JSONResponse(status_code=200, content=content)

        else:
            return JSONResponse(status_code=503)
    except Exception as e:
        state['review_start'] = False
        line(e)
        print(e)
        return JSONResponse(status_code=500)

@app.get('/review/parameters', include_in_schema=True)
def metadata(project_name: str, measurement_name: str):
    try:
        global metadataCollection
        _ID_FILTER = 0
        ret = metadataCollection.find({'project': project_name, 'measurement': measurement_name}, {
                                      "_id": _ID_FILTER}).limit(1).sort({"$natural": -1})
        # ret = metadataCollection.find_one(
        #     {'project': project_name, 'measurement': measurement_name}, {"_id": _ID_FILTER})
        return JSONResponse(status_code=200, content=ret)
    except Exception as e:
        return JSONResponse(status_code=500)


@app.post('/traverse', include_in_schema=False)
def set_trav(current_user: User = Depends(get_current_active_user)):
    pass


@app.get("/live-data", include_in_schema=False)
def live_data(q: Optional[str] = None):
    statsData, numHist, volHist, bins = get_stats(
        diameter=diameters, binning="auto", hist_max=None)
    histData = [bins[1:].tolist(), numHist.tolist(), volHist.tolist()]
    if q == 'histogramData':
        to_send = [histData]
    elif q == 'statsData':
        to_send = [statsData]
    else:
        to_send = [histData, statsData]
    return Response(content=to_send, media_type="application/json")


@app.get('/traverse', include_in_schema=True)
def traverse():
    content = {'available': False}
    return JSONResponse(status_code=200, content=content)


@app.get('/data')
def poll_data(bin_size: Optional[int] = None, clear_data: Optional[bool] = None):

    global ph, has_cam, stats, hist

    # ------- function just for testing ------
    try:
        if clear_data is not None:
            ph.reset_stats()
            return JSONResponse(status_code=200)

        else:
            if has_cam:
                is_test = True
            else:
                is_test = True

            if is_test == True:
                Nd = int(3000 + np.random.rand()*500)
                d = np.random.randn(Nd)
                d = d + np.abs(d.min())*1.05
                ph.diameters = d/d.max()*150.
                stats, hist = ph.get_stats(bin_size=5)
            else:
                if bin_size is not None:
                    bin_size = bin_size
                stats, hist = ph.get_stats(bin_size=bin_size)

            content = {}
            content['stats'] = stats
            content['hist'] = hist
            content['review_stats'] = review_stats
            content['review_hist'] = review_hist

        return JSONResponse(status_code=200, content=content)
    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.get('/projects')
# get projects details?
def get_proj(project_name: Optional[str] = None):
    global projectsCollection, mazliteDB, mongoClient
    if project_name is None:
        ret = list(projectsCollection.find({}, {'_id': 0}))
    else:
        ret = list(projectsCollection.find(
            {"project_name": project_name}, {'_id': 0}))
    return ret


@app.post('/projects')
# change to project
def set_proj(item: Data):
    global projectsCollection, mazliteDB, mongoClient, ph
    try:
        pname = item.dict().get('project_name')
        logging.debug(item.dict())
        _ID_FILTER = 0
        check = projectsCollection.find(
            {'project_name': pname}, {"_id": _ID_FILTER})
        if check.count() > 0:
            projectsCollection.delete_many({'project_name': pname})
        ret = projectsCollection.insert(item.dict())
        ph.create_project(pname)
        return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        logger.error(e)
        return JSONResponse(status_code=500)


@app.put('/projects')
# change to project
def _change_proj(item: dict = None, old_name: Optional[str] = None, new_name: Optional[str] = None):
    global projectsCollection, mazliteDB, mongoClient, measurementsCollection, ph
    try:
        if new_name is None:
            new_name = item.get('name')
        if old_name != new_name:
            # deal with renaming project
            pname = old_name
            _ID_FILTER = 1
            if old_name is not None:
                pname = old_name
            check = projectsCollection.find(
                {'project_name': pname}, {"_id": _ID_FILTER})
            if check.count() > 0:

                # get project id and rename name directly
                for d in check:
                    if type(d) == dict:
                        project_id = d['_id']
                projectsCollection.update({'_id': project_id}, {
                    '$set': {'project_name': new_name}})

                # rename project in all measurements collections
                bulk = measurementsCollection.initialize_unordered_bulk_op()
                bulk.find({'project_name': pname}).update(
                    {'$set': {'project_name': new_name}})
                bulk.execute()
                ph.rename_project(old_name=old_name, new_name=new_name)

        # end dealing with renaming
        else:
            # deal with edit project metadata
            pname = old_name
            for k, v in item.get('params').items():
                # print(k, v)
                sub = '{}.{}'.format('params', k)
                for i in v:
                    measurementsCollection.update(
                        {"project_name": pname}, {"$addToSet": {sub: i}})
                    projectsCollection.update({"project_name": pname}, {
                        "$addToSet": {sub: i}})

        return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        logger.error(e)
        return JSONResponse(status_code=500)


@app.delete('/projects')
def _delete_proj(project_name: str = None):
    global projectsCollection, mazliteDB, mongoClient, measurementsCollection, ph
    if not project_name:
        return JSONResponse(status_code=500)

    try:
        pname = project_name
        check = projectsCollection.find(
            {'project_name': pname}, {"_id": _ID_FILTER})
        if check.count() > 0:
            projectsCollection.delete_many({'project_name': pname})
        check = measurementsCollection.find(
            {'project_name': pname}, {"_id": _ID_FILTER})
        if check.count() > 0:
            measurementsCollection.delete_many({'project_name': pname})
        try:
            ph.delete_project(project_name)
        except:
            pass
        return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        logger.error(e)
        return JSONResponse(status_code=500)


@app.get('/measurements')
def _get_measurements(project_name: Optional[str] = None, measurement_name: Optional[str] = None, record: Optional[str] = None):
    global measurementsCollection, mazliteDB, mongoClient
    ret = None
    if not record:
        if project_name is None and measurement_name is None:
            raise HTTPException(
                status_code=404, detail="Project name is not specified")
        if not measurement_name:
            ret = list(measurementsCollection.find(
                {'project_name': project_name}, {'_id': _ID_FILTER}))
        else:
            ret = list(measurementsCollection.find(
                {'project_name': project_name, 'measurement_name': measurement_name}, {'_id': _ID_FILTER}))
        return ret
    else:
        ret = list(measurementsCollection.find(
            {'record': record}, {'_id': _ID_FILTER}))
        return ret


@app.post('/measurements')
def _set_measurements(item: Optional[model_MEASUREMENT], record: Optional[str] = None):
    global measurementsCollection, mazliteDB, mongoClient, ph
    ret = None
    _item = item.dict()
    try:
        if not record:
            if 'project_name' in _item and 'measurement_name' in _item:
                _item = set_record(_item)
                ret = measurementsCollection.insert(_item)
                ph.create_measurement(
                    _item.get('project_name'), _item.get('measurement_name'))
                return {'record': _item.get('record')}
            else:
                raise Exception
        else:
            _item['record'] = record
            ret = measurementsCollection.insert(_item)
            ph.create_measurement(project_name, measurement_name)
            return {'record': _item.get('record')}

    except Exception as e:
        logger.error('Error: {}'.format(e))
        raise HTTPException(
            status_code=err.get('ERR'), detail="Server Error.")


@app.delete('/measurements')
def _delete_measurement(project_name: str, measurement_name: str, record: Optional[str] = None):
    global measurementsCollection, mazliteDB, mongoClient, ph
    try:
        _ID_FILTER = 0
        check = measurementsCollection.find(
            {'project_name': project_name, 'measurement_name': measurement_name})
        if check.count() > 0:
            check = measurementsCollection.find(
                {'project_name': project_name, 'measurement_name': measurement_name})
            measurementsCollection.delete_many(
                {'project_name': project_name, 'measurement_name': measurement_name})

            ph.delete_measurement(project_name, measurement_name)
        return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        logger.error(e)
        return JSONResponse(status_code=500)


# @app.get('/storage')
def _get_strg_info():
    global dataRoot
    mega = 2**20
    camera_img_size = 10
    hdds = {}
    data, main = dataRoot, '/'
    data_info = psutil.disk_usage(data)
    # main_info = psutil.disk_usage(main)
    numpics = data_info.free/(mega)
    numpics = 0.95 * numpics
    numpics = numpics // camera_img_size
    hdds['num_pics_remaining'] = numpics  # count
    hdds['total'] = str(data_info.total/(2**40))[0:3]
    hdds['free'] = str(data_info.free/(2**40))[0:3]
    return hdds


@app.get('/video')
# def disk_stream(file: str = None, measurement_name: str = None, project_name: str = None):
def disk_stream():
    global dataRoot, video_folder, video_path
    if video_path == '':
        f = 'video.mp4'
    elif video_path != '':
        f = video_path
    try:
        file_like = open(f, mode="rb")
        return StreamingResponse(file_like, media_type="video/mp4")
    except:
        return JSONResponse(status_code=500)


@app.put('/measurements')
def _change_measurement(item: Optional[model_MEASUREMENT], record: Optional[str] = None):
    global measurementsCollection, mazliteDB, mongoClient
    _item = item.dict()
    try:
        check = None
        if record is None:
            raise Exception
        else:
            check = measurementsCollection.find({'record': record})
            if check.count() > 0:
                measurementsCollection.delete_many({'record': record})

            _item['record'] = record
            check = measurementsCollection.insert(_item)
            return JSONResponse(status_code=200)

    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500)


@app.get("/setting", include_in_schema=False)
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/update")
async def update(request: Request):
    return JSONResponse(status_code=200, content={'msg': 'Your software is up to date.'})


@app.get("/help")
async def read_item(request: Request):
    # try:
    global machineCollection, ins
    ret = machineCollection.find_one({}, {'_id': 0})
    if ret is not None:
        return JSONResponse(status_code=200)

    if not ins:

        try:
            url = 'http://localhost:9990/flash'
            r = requests.get(url=url)
            r = r.json()
            ins['flash_settings'] = r
        except Exception as e:
            print(e)
            line(e)
            pass

    try:
        url = 'http://localhost:9999/heartbeat'
        r = requests.get(url=url)
        r = r.json()

        ret['MachineInfo']['MACAddress'] = r.get('MAC address')
        ret['MachineInfo']['IPAddress'] = r.get('IP address')
        process = subprocess.check_output(["git", "rev-parse","--short","HEAD"])
        git_head_hash = process.decode('utf-8').strip()

        ret['MachineInfo']['SoftwareVersion'] = git_head_hash

        ret['MachineSpec']['StorageCap'] = _get_strg_info().get('total')
        ret['MachineSpec']['NumberPictures'] = _get_strg_info().get(
            'num_pics_remaining')

        ret['SystemDiagnostics']['CameraTemp'] = '65' #get this from camera
        ret['SystemDiagnostics']['LaserTemp'] = ins.get(
            'flash_settings').get('device_temp')
        ret['SystemDiagnostics']['LaserControllerTemp'] = ins.get(
            'flash_settings').get('controller_temp')
        ret['SystemDiagnostics']['CPUTemp'] = r.get(
            'CPU Temprature').split('C')[0]
        ret['SystemDiagnostics']['CPULoad'] = r.get('CPU Usage').split('%')[0]
        ret['SystemDiagnostics']['MemoryUsage'] = r.get('Memory Usage')
        ret['SystemDiagnostics']['Uptime'] = r.get('Uptime')
        ret['SystemDiagnostics']['StorageAvailable'] = _get_strg_info().get('free')

    except Exception as e:
        line(e)
        print(e)
        ret = backup

    return JSONResponse(status_code=200, content=ret)
    # except:
    #     return JSONResponse(status_code=500)


@app.get("/settings")
def get_usersettings(name: Optional[str] = None):
    global userPreferenceCollection, mazliteDB, mongoClient
    try:
        if not name:
            ret = userPreferenceCollection.find({}, {'_id': 0})
            name_vec = []
            for i in ret:
                if type(i.get('name')) is str:
                    name_vec.append(i.get('name'))
            content = {}
            content['available_profiles'] = name_vec
            return JSONResponse(content=content, status_code=200)

        else:
            ret = userPreferenceCollection.find({"name": name}, {'_id': 0})
            if ret.count() > 0:
                content = list(ret)[0]
            else:
                content = {}
            # if len(content) == 1:
            #     content = content[0]
            return JSONResponse(content=content, status_code=200)
    except Exception as e:
        line(e)
        return JSONResponse(content={}, status_code=500)


@app.post('/settings')
def set_usersettings(item: dict):
    global userPreferenceCollection, mazliteDB, mongoClient
    try:
        append = ''
        name = item.get('name')
        ret = userPreferenceCollection.find({"name": name}, {'_id': 0})
        # if ret.count() > 0 and item.get('name').lower() != 'default':
        #     item['name'] = name
        if not old_name:
            ret = userPreferenceCollection.delete_many(
                {"name": name}, {'_id': 0})
            ret = userPreferenceCollection.insert(item)
        else:
            ret = userPreferenceCollection.delete_many(
                {"name": old_name}, {'_id': 0})
            ret = userPreferenceCollection.insert(item)

        return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.put('/settings')
def change_usersettings(item: dict, old_name: str):
    global userPreferenceCollection, mazliteDB, mongoClient
    try:
        name = item.get('name')
        if old_name.lower() != 'Mazlite_Default':

            ret = userPreferenceCollection.delete_many(
                {"name": old_name})

            ret = userPreferenceCollection.insert(item)

            return JSONResponse(status_code=200)
        # raise Exception
    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.delete('/settings')
# change to project
def delete_usersettings(name: str):
    global userPreferenceCollection, mazliteDB, mongoClient
    try:
        if "default" not in name.lower():
            ret = userPreferenceCollection.delete_many(
                {"name": name})
            return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.get('/status')
def get_status():
    global state, has_cam
    content = {}

    if not has_cam:
        rn = int(np.round(np.random.rand()*7))
        clr, fs, msg = _get_status_message(rn)
        light_status = {'colour': clr, 'flash_state': fs, 'msg': msg}
        content['lights'] = light_status
        return JSONResponse(status_code=200, content=content)
    else:
        light_status = {
            'review':
            {
                'colour': 'Blue',
                'flash_state': 'Blinking Fast' if state.get('review_start') is True else 'Solid',
                'msg': 'The device is available for the review process.' if state.get('review_start') is False else 'The device is performing a review process.'
            },
            'capture':
            {
                'colour': 'Green',
                'flash_state': 'Blinking Fast' if state.get('capture_start') is True else 'Solid',
                'msg': 'The device is available for measurement.' if state.get('capture_start') is False else 'The device is capturing and processing.'
            },
            'operation_fail':
            {
                'colour': 'Yellow',
                'flash_state': 'Blinking' if state.get('failed_op') is True else 'Off',
                'msg': 'There are no issues.' if state.get('failed_op') is False else 'The device has failed one or more operations.'
            },
            'failed':
            {
                'colour': 'Red',
                'flash_state': 'Blinking' if state.get('failed_badly') is True else 'Off',
                'msg': 'There are no issues.' if state.get('failed_badly') is False else 'The device is in critical condition and needs immediate attention. Perform a restart.'
            }
        }
        content['review'] = state.get('review_start')
        content['capture'] = state.get('capture_start')

        content['lights'] = light_status
        return JSONResponse(status_code=200, content=content)


@app.post("/batch")
async def create_upload_file(project_name: str, uploadedFile: UploadFile = File(...)):
    """
    Upload spreadsheet
    """
    try:
        global dataRoot, ph, measurementsCollection
        name = '{} - systematic measurement upload'.format(project_name)
        ext = 'xlsx'
        file_location = ph._get_subfolder_path('xlsx', project_name)

        # read csv data
        parsed_data = msu.parse_systematic_csv(uploadedFile.file)
        print(parsed_data)
        for k, v in parsed_data.items():
            item = {}
            item['measurement_name'] = str(k)
            item['project_name'] = str(project_name)
            item['params'] = v
            item = set_record(item)
            ret = measurementsCollection.insert(item)
            ph.create_measurement(
                item['project_name'], item['measurement_name'])

        if os.path.isdir(file_location):
            file_name = "{loc}/{name}.{ext}".format(
                loc=file_location, name=name, ext=ext)
            with open(file_name, "wb+") as file_object:
                shutil.copyfileobj(uploadedFile.file, file_object)
            return JSONResponse(status_code=200)

    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.get('/manual')
def get_manual():
    """
    Download machine user manual.
    """
    try:
        filepath = './misc/manual.pdf'
        if os.path.isfile(filepath):
            return FileResponse(path=filepath, filename='mazlite_user_manual.pdf', media_type='application/pdf')
        return JSONResponse(status_code=404)
    except:
        return JSONResponse(status_code=500)


@app.get('/batch')
def download(measurement_name: Optional[str] = None, project_name: Optional[str] = None):
    """
    Download template.
    """
    try:
        global dataRoot
        name = 'systemic'
        ext = 'xlsx'
        sub_folder = 'xlsx'
        if not measurement_name or not project_name:
            filepath = './misc/systemic.xlsx'
            if os.path.isfile(filepath):
                return FileResponse(path=filepath, filename='measurement_template.xlsx', media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        else:
            filepath = "{}/{}/{}/{}/{}.{}".format(dataRoot,
                                                  project_name, sub_folder, measurement_name, name, ext)
        if os.path.isfile(filepath):
            return FileResponse(path=filepath, filename='{}_{}_{}.xlsx'.format(project_name, measurement_name, 'measurement'), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        else:
            return JSONResponse(status_code=404)
    except:
        return JSONResponse(status_code=500)


@app.get('/stop')
def stop_review():
    try:
        global savingImages, img_array, img_id, imgv, state, capture_settings
        img_array.clear()
        img_id.clear()
        imgv.clear()
        savingImages = 0
        while savingImages != 0:
            sleep(0.05)
        state['capture_start'] = False
        return JSONResponse(status_code=200)
    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.get('/start')
def start(project_name: str, measurement_name: str):
    b = time.time()
    global _proj, _meas, savingImages, img_array, img_id, imgv, ph
    global state, has_cam, capture_settings, operation_settings, stream_settings
    global operation_settings, stats, hist, state, metadataCollection
    global FMT, ins, timeout, duration

    if state.get('capture_start') is False:
        state['capture_start'] = True
        img_array.clear()
        img_id.clear()
        imgv.clear()

        _meas = measurement_name
        _proj = project_name

        ph.initialize_pipeline(project_name=_proj, measurement_name=_meas)
        savingImages = capture_settings.get('numberImages')

        if has_cam:

            try:
                if operation_settings.get("capture_mode").lower() == 'sequential':
                    print('seq')
                    while savingImages != 0:
                        sleep(0.1)
                    img_array = [x.img for x in imgv]
                    img_id = [x.name for x in imgv]
                    # print('processing {} images'.format(len(img_array)))
                    ss = time.time()
                    ph.pipeline(do_process=capture_settings.get('detect_particles'), do_save=capture_settings.get(
                        'save_images'), img_array=img_array, img_id=img_id)
                    logging.debug('image proc took {}'.format(time.time()-ss))
                    if capture_settings.get('detect_particles'):
                        stats, hist = ph.get_stats()
                        # ph.save_stats(_proj, _meas, stats, hist)
                    state['capture_start'] = False

                   # while savingImages != 0:
                   #     img_array = [x.img for x in imgv]
                   #     img_id = [x.name for x in imgv]
                   #     ph.pipeline(do_process=capture_settings.get('detect_particles'), do_save=capture_settings.get(
                   #         'save_images'), img_array=img_array, img_id=img_id)
                   # stats, hist = ph.get_stats()
                   # ph.save_stats(_proj, _meas, stats, hist)
                   # state['capture_start'] = False

                else:
                    while savingImages != 0:
                        sleep(0.1)
                    img_array = [x.img for x in imgv]
                    img_id = [x.name for x in imgv]
                    # print('processing {} images'.format(len(img_array)))
                    ss = time.time()
                    ph.pipeline(do_process=capture_settings.get('detect_particles'), do_save=capture_settings.get(
                        'save_images'), img_array=img_array, img_id=img_id)
                    logging.debug('image proc took {}'.format(time.time()-ss))
                    if capture_settings.get('detect_particles'):
                        stats, hist = ph.get_stats()
                        ph.save_stats(_proj, _meas, stats, hist)
                    state['capture_start'] = False

                ins['capture_settings'] = capture_settings
                ins['operation_settings'] = operation_settings
                ins['stream_settings'] = stream_settings
                ins['project'] = _proj
                ins['measurement'] = _meas
                ins['time'] = datetime.datetime.now().strftime(FMT).format()
                ins['help_settings'] = help_settings

                try:
                    url = 'http://localhost:9990/flash'
                    r = requests.get(url=url)
                    ins['flash_settings'] = r.json()

                except Exception as e:
                    line(e)
                    ins['flash_settings'] = {}

                # ret = metadataCollection.delete_many(
                #     {'project': _proj, 'measurement': _meas})

                ret = metadataCollection.insert(ins)
                e=b-time.time()
                duration = e
                return JSONResponse(status_code=200)
            except Exception as e:
                state['capture_start'] = False
                line(e)
                return JSONResponse(status_code=500)
        else:
            state['capture_start'] = False
            return JSONResponse(status_code=200, content={})
    else:
        return JSONResponse(status_code=503)


@app.get('/ping', include_in_schema=False)
# change to post
async def ping():
    return JSONResponse(status_code=200)


@app.get('/reboot', include_in_schema=False)
# change to post
async def reboot(op: Optional[str] = None):
    if not op:
        os.system('sudo reboot')
    else:
        os.system('sudo poweroff')


@app.get('/videoStreamSettings')
def video_stream_info():
    try:
        global video_stream_info_settings
        return JSONResponse(status_code=200, content=video_stream_info_settings)
    except:
        return JSONResponse(status_code=500)


@app.get('/streamSettings')
def getStreamSettings():
    # todo: read db
    global sm,duration
    try:
        global stream_settings
        _duration = str(timedelta(seconds=duration))
        stream_settings['duration'] = _duration
        stream_settings['fps']=sm.get_current_thread().getFPS().value
        return JSONResponse(status_code=200, content=stream_settings)
    except Exception as e:
        line(e)
        print(e)
        return JSONResponse(status_code=500)


@app.post('/streamSettings')
def setStreamSettings(markParticles: Optional[bool] = None, autoBoost: Optional[bool] = None, downsample: Optional[int] = None, broadcastFPS: Optional[int] = None, play: Optional[bool] = None):
    # todo: write db
    try:
        global stream_settings

        if markParticles is not None:
            stream_settings['mark'] = markParticles
        if autoBoost is not None:
            stream_settings['boost'] = autoBoost
        if downsample is not None:
            stream_settings['downsample']['ratio'] = downsample
        if broadcastFPS is not None:
            stream_settings['broadcastFPS'] = broadcastFPS
        if play is not None:
            stream_settings['play_state'] = play
        # print(stream_settings)
        return JSONResponse(status_code=200)
    except Exception as e:
        line(e)
        return JSONResponse(status_code=500)


@app.get('/captureSettings')
def getCaptureSettings():
    global capture_settings
    try:
        return JSONResponse(status_code=200, content=capture_settings)
    except:
        return JSONResponse(status_code=500)


@app.post('/captureSettings')
def setCaptureSettings(cap: dict):
    global capture_settings
    # capture_settings = {'profile': 'default', 'num_pics_to_take': 50,
    #                     'num_particles_to_detect': 0, 'detect_particles': False, 'save_images': False}
    try:
        # get profile from DB
        capture_settings['profile'] = cap.get('profile')
        capture_settings['numberImages'] = cap.get('numberImages')
        # set number of images to take
        capture_settings['detect_particles'] = cap.get('detectParticles')
        # set detectparticles in dh
        capture_settings['save_images'] = cap.get('saveImages')
        # set save image in dh
        return JSONResponse(status_code=200)

    except:
        line(e)
        return JSONResponse(status_code=500)


@app.get('/operationSettings')
def setOperationSettings(measurementMode: Optional[str] = None, captureMode: Optional[int] = None, scanMode: Optional[bool] = None, measurementType: Optional[bool] = None, triggerMode: Optional[str] = None):
    global operation_settings
    try:
        return JSONResponse(status_code=200, content=operation_settings)
    except:
        return JSONResponse(status_code=500)


@app.post('/operationSettings')
def setOperationSettings(caps: dict, measurementMode: Optional[str] = None, captureMode: Optional[int] = None, scanMode: Optional[bool] = None, measurementType: Optional[bool] = None, triggerMode: Optional[str] = None):
    global operation_settings
    operation_settings['measurementMode'] = caps.get('particle')
    operation_settings['capture_mode'] = caps.get('captureMode')
    operation_settings['scanMode'] = caps.get('scanMode')
    operation_settings['measurementType'] = caps.get('measurementType')
    operation_settings['triggerMode'] = caps.get('triggerMode')

    try:
        return JSONResponse(status_code=200)

    except:
        line(e)
        return JSONResponse(status_code=500)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    global ph, has_cam, stats, hist
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # await manager.send_personal_message(f"You Got: {data}", websocket)
            await manager.broadcast({data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left")

try:
    sm = SystemMode.get_instance()
    sm.clear_all_modes()
    maincam = Camera.GetInstance()
    maincam.LoadIniFile(None)
    sm.change_mode_to_streaming()
    sm.get_current_thread().frame_event += prepareForStreaming
    sm.start_stream_thread()

except:
    has_cam = False
