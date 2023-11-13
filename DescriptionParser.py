import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from nltk import pos_tag, tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


from timeit import default_timer as timer

def run_clean_dataset(df,desc_orig,desc_output):

    unspsc_words = pd.read_csv('wordslist.csv')
    #delete this line for power bi
    unspsc_words = unspsc_words['words'].to_list()
    stop_words =  set(stopwords.words('english'))

    keep_words = set(['heatshrink','thinwall','heatsink', 'photohelic','measurometer','motherboard','shockwave','quickstart',
    'keypad','ethernet','nitril','interconnecting','multitach','riveted','collett','compumotor','consistancy','knockouts',
    'magnavalve','modbus','hydramotor','setpoint','amptector','x-ray','vuc','photoeye','ultrameter','bargraph','recalibrator','bandwidth',
    'axis','dosimeter','variation','voltmeter','intercom','exciter','smartups','dvc','alkyd'])

    acronyms = set(['ac','v','plc','rfi','lub','dc','pcb','cpu','xray','oring','cvc'])

    fix_dict = {'cyl':'cylinder','pc':'computer','nyl':'nylon','mb':'board','mod':'module','ctrl':'control','shldd':'shielded','plg':'plug','lgc':'logic','rly':'relay','srv':'servo','pwr':'power',
    'impct':'impact','rollr':'roller','brg':'bearing','pneum':'pneumatic','torq':'torque','brgs':'bearing','swtch':'switch','pusbttn':'pushbutton','serral':'serial','conctn':'connector',
    'extsn':'extension','intfc':'interface','thrmcpl':'thermocouple','thrmstt':'thermostat','trnmtr':'transmitter','cntct':'contact','flngd':'flanged','spher':'spherical','sphrcl':'spherical','thrst':'thrust',
    'nitril':'nitrile','pwrgrip':'power grip','cabel':'cable','ballasat':'ballast','ehernet':'ethernet','prxmty':'proximity','reflctv':'reflective','staright':'straight','fbroptc':'optic fiber','fiberoptic':'optic fiber',
    'photoelc':'photoelectric','plstc':'plastic','dsply':'display','strgt':'straight','recptng':'receptacle','recpt':'receptacle','crclr':'circular','insltd':'insulated','revrsing':'reversing','adptr':'adapter',
    'semcdtr':'semiconductor','conctr':'connector','receptacl':'receptacle','indtve':'inductive','retrcbl':'retractable','cndctrs':'conductor','redcg':'reducing','pnmtc':'pneumatic',
    'bd':'board','snstv':'sensitive','comprsd':'compressed','connecctor':'connector','recvr':'receiver','comprsr':'compressor','dspble':'disposable','diaph':'diaphragm','diffrntl':'differential',
    'recucer':'reducer','gearr':'gear','flshlgt':'flashlight','fttng':'fitting','plwblk':'pillow block','tst':'test','conctg':'connector','genrlprps':'general purpose',
    'dirctnl':'directional','cntfgl':'centrifugal','mechncl':'mechanical','dirve':'drive','solidst':'solid state','emrgncy':'emergency','wlmnt':'wall mount','cyldrcl':'cylindrical',
    'excitr':'exciter','vacumm':'vacuum','cbl':'cable','shld':'shield','assy':'assembly',
    'miodule':'module','det':'detector','curr':'current','diff':'differential','dir':'direct','elect':'electric','comm':'communication','brk':'brake','tach':'tachometer','freq':'frequency',
    'ctl':'control','genera':'genrator','drv':'drive','accel':'accelerator','temp':'temperature','xfmr':'current transformer','xmtr':'transmitter',
    'scr':'sillicon controlled rectifier','proc':'processor','pnl':'panel','pres':'pressure','intf':'interface','ct':'current transformer','gra':'graphics','mem':'memory','rect':'rectangular','devicenet':'device net',
    'ic':'integrated circuit','eps':'insulated','controld':'control','vl':'molded case circuit breaker','sw':'switch','asi':'analog module','rel':'relay','resis':'resistant',
    'circ':'circuit','fu':'fuse','mcb':'miniature circuit breaker','drve':'drive','reg':'regulator','rec':'recorder','vibr':'vibrator','sma':'smart','contr':'contactor','presuure':'pressure','mtr':'motor',
    'actionator':'actuator','hydrolic':'hydraulic','pls':'control','lvr':'lever','byps':'bypass','splitcore':'split core','spd':'surge protection device','rcvr':'receiver','fngr':'finger','clstr':'cluster',
    'wl':'circuit breaker','fluor':'fluorescent','dig':'digital','digit':'digital','prog':'programmable','conn':'connector','abox':'box','panelmate':'display','quickpanel':'display',
    'sas':'hard drive','sata':'hard drive','touchpanel':'display','touchglass':'display','touchscreen':'display','mobileview':'display','panelview':'display','touchmonitor':'display',
    'repl':'replication','intl':'internal','hmi':'interface','softstart':'solid state relay','stater':'starter','baord':'board','profibus':'signal converter','curatin':'curtain','gage':'gauge',
    'modul':'module','weigh':'weight','var':'variable','mulimeter':'multimeter','bearings':'bearing','filt':'filter','transister':'transistor','inv':'invert','brd':'board','convectron':'vacuum gauge',
    'propel':'propellant','diagraph':'diaphragm','ecs':'electronic control suspension','lightcurtain':'light curtain','nutdriver':'nut driver','servodrive':'servo drive','safegrip':'safe grip',
    'powerstation':'power station','handcleaner':'hand cleaner','flowgauge':'flowmeter','compresor':'compressor','duracell':'batteries','microswitch':'switch','earplugs':'ear plug','gripbelt':'grip belt',
    'fusebolt':'fuse bolt','powergrip':'power grip','powerhandler':'power handler','linkbelt':'link belt','halflink':'half link','gearhead':'gear head','gearmotor':'gear motor','barcode':'bar code',
    'processboard':'processor board','starters':'starter','microstep':'micro step','servos':'servo','vibrameter':'vibration meter','weigher':'weight scale','uv':'ultraviolet','powersupply':'power supply',
    'servoamp':'servo amplifier','servomotor':'servo motor','acdrive':'drive ac','pmate':'display','chasis':'chassis','crt':'display','panelveiw':'display','servovalve':'servo valve',
    'controler':'controller','kypd':'keypad','tscrn':'display','stepdrive':'step drive','displaywi':'display','sevo':'servo','convertor':'converter','applifier':'amplifier','printhead':'printer head',
    'montior':'display','megohmeter':'megohmmeter','scopemeter':'oscilloscope','contoller':'controller','modlue':'module','reciever':'receiver','xmtter':'transmitter','ocilliscope':'oscilloscope',
    'rackmount':'rack mount','trasmitter':'transmitter','guage':'gauge','smc':'solid state relay','hydradulic':'hydraulic','cntrl':'control','acuator':'actuator','hyd':'hydraulic','solniod':'solenoid',
    'pumo':'pump','ctrlr':'controller','scrn':'screen','versaview':'display','pneu':'pneumatic','dect':'detector','powewr':'power','multitouch':'display','micrologix':'plc','sew-eurodrive':'variable speed drives',
    'eurodrive':'variable speed drives','pwm':'bridge inverter','powerflex':'variable speed drive','cpu':'plc module','brak':'brake','dgtl':'digital','encl':'enclosure','expnsn':'expansion','psi':'pressure','tmedly':'time delay',
    'otp':'output','fltr':'filter','lubrctr':'lubricator','string':'thermistor','recvr':'receiver','revrsing':'reversing','mgntc':'magnetic','tpr':'taper','adh':'adhesive','cvall':'coverall','brigeport':'bridgeport','mach':'machine'}


    words_to_remove = set()

    oem_names = set(['b&b electronics','schneider electric','liquid solids','electric co','advanced micro controls','cuttler hammer','cutler hammer','cutler-hammer',
    'general electric','moisture systems','spectrum controls','spectrum controls','switching systems''superior electric','control technology','flash tech','square d',
    'balance dynamics','long beach','bodine electric','diamond systems','snap-pac','pacific scientific','ge controls','scan-core','union carbide','rapid-air','cleaver brooks',
    'vee-arc','star delta','generic automation','electro-craft','rockwell automation','bei electronics','control techniques','immersion corporation','electronics inc','landis tool',
    'anaheim automation','electrical south','electric system inc','seco tach','total control','electron-machine','hughes aircraft','armor guardlogix',
    'fuji electric','acme','oliver','danner','muck','frontier','co-cool','steelox','sew-eur'
    ])

    not_change = set(['bearing','led','flanged','bushing'])

    short_words_keep= set(['led','pcb','cpu','box','fan','lcd','ups','usb','ups','plc','can'])


    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wn.ADJ
        elif nltk_tag.startswith('V'):
            return wn.VERB
        elif nltk_tag.startswith('N'):
            return wn.NOUN
        elif nltk_tag.startswith('R'):
            return wn.ADV
        else:         
            return None

    def remov_duplicates(input):
        input = input.split(" ")
        for i in range(0, len(input)):
            input[i] = "".join(input[i])
        UniqW = Counter(input)
        s = " ".join(UniqW.keys())
        return s

    def removeStopWords(string):
        word_tokens = word_tokenize(string)
        filtered_description = []
        for w in word_tokens:
            if w not in stop_words and w not in ['can'] and w not in ['model']:
                if w in fix_dict:
                    filtered_description.append(fix_dict.get(w))
                else:
                    filtered_description.append(w)
        filtered_description = ' '.join(map(str,filtered_description))
        return filtered_description

    def removeWords(string):
        string = string.split()
        for w in string:
            if w not in keep_words and w not in short_words_keep and w not in acronyms and w not in unspsc_words:   
                words_to_remove.add(w)
        

    def clean_description(string):
        string = re.sub(r'\(.*\)',' ',string)
        string = re.sub(r'[^\w\s]',' ',string)
        string = re.sub(r'pcs','',string)
        string = re.sub(r'1 ph|1ph','single phase ',string)
        string = re.sub(r'[2-9] ph|[2-9]ph','multi phase ',string)
        string = re.sub(r'\d+mb','',string)
        string = re.sub(r'\d+\smb','',string)
        string = re.sub(r'\d+pc','',string)
        string = re.sub(r'pc\d+','',string)
        string = re.sub(r'\d+dc','dc',string)
        string = re.sub(r'\d+v','',string)
        string = re.sub(r'v\d+','',string)
        string = re.sub(r'\d+ac','ac',string)
        string = re.sub(r'\d+','',string)
        string = re.sub(r'ii+','',string)
        string = re.sub(r'\b[a-uw-z]\b|\b [a-uw-z]\b','',string)
        string = re.sub(r'\s+',' ',string)
        return string

    pat = r'\b(?:{})\b'.format('|'.join(oem_names))
    df[desc_output] = df[desc_orig]
    df[desc_output] = df[desc_output].fillna(value='empty')
    df[desc_output] = df[desc_output].apply(lambda x: x.encode('ascii',errors='ignore').decode()).str.lower()
    df[desc_output] = df[desc_output].str.replace(pat,'',regex=True)
    df[desc_output] = df[desc_output].str.replace(' o2 ',' oxygen ',regex=False)
    df[desc_output] = df[desc_output].str.replace('x-ray','xray',regex=False)
    df[desc_output] = df[desc_output].str.replace('x ray','xray',regex=False)
    df[desc_output] = df[desc_output].str.replace('q-panel',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('micro logix',' plc ',regex=False)
    df[desc_output] = df[desc_output].str.replace('panel mate',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('industrial screen',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('power flex',' variable speed drive ',regex=False)
    df[desc_output] = df[desc_output].str.replace('flex power',' variable speed drive ',regex=False)
    df[desc_output] = df[desc_output].str.replace('rail way',' railwway ',regex=False)
    df[desc_output] = df[desc_output].str.replace('way rail',' railwway ',regex=False)
    df[desc_output] = df[desc_output].str.replace('o-ring',' oring ',regex=False)
    df[desc_output] = df[desc_output].apply(clean_description)
    df[desc_output] = df[desc_output].apply(lambda x: x if len(x) < 183 else 'empty')
    df[desc_output] = df[desc_output].apply(removeStopWords)
    df[desc_output] = df[desc_output].str.replace('quick panel',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('panel view',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('view panel',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('panel screen',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('screen panel',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('versa view',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('work station',' workstation ',regex=False)
    df[desc_output] = df[desc_output].str.replace('push button',' pushbutton ',regex=False)
    df[desc_output] = df[desc_output].str.replace('power flex',' variable speed drive ',regex=False)
    df[desc_output] = df[desc_output].str.replace('panel quick',' display ',regex=False)
    df[desc_output] = df[desc_output].str.replace('station work',' workstation ',regex=False)
    df[desc_output] = df[desc_output].str.replace('button push',' pushbutton ',regex=False)
    df[desc_output] = df[desc_output].str.replace('start',' starter ',regex=False)
    df[desc_output] = df[desc_output].str.replace('connection',' connector ',regex=False)
    df[desc_output] = df[desc_output].str.replace('violet ultra',' ultraviolet ',regex=False)
    df[desc_output] = df[desc_output].str.replace('ultra violet',' ultraviolet ',regex=False)
    df[desc_output] = df[desc_output].str.replace('starter soft',' solid state relay ',regex=False)
    df[desc_output] = df[desc_output].str.replace('soft starter',' solid state relay ',regex=False)
    df[desc_output] = df[desc_output].str.replace('ink jet',' inkjet ',regex=False)
    df[desc_output] = df[desc_output].str.replace('\s+', ' ',regex=True)
    df[desc_output].apply(removeWords)
    pat2 = r'\b(?:{})\b'.format('|'.join(words_to_remove))
    df[desc_output] = df[desc_output].str.replace(pat2,' ',regex=True)
    df[desc_output] = df[desc_output].str.replace('\s+', ' ',regex=True)

    for index, description in enumerate(df[desc_output]):
        final_words = []
        word_lemmatized = WordNetLemmatizer()
        pos_tagged = pos_tag(word_tokenize(description))
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        for word, tag in wordnet_tagged:
            if word in not_change:
                final_words.append(word)
            else:
                if tag is None:
                    final_words.append(word)
                elif tag not in ['DT','PRP','MD','TO','PRP','CC']:       
                    final_words.append(word_lemmatized.lemmatize(word, tag))
        final_words = list(dict.fromkeys(final_words))
        final_words =' '.join(map(str,final_words))
        df.loc[index,desc_output] = final_words
    df[desc_output] = df[desc_output].str.replace('\s+', ' ',regex=True)
    df[desc_output] = df[desc_output].apply(remov_duplicates)
    df[desc_output] = df[desc_output].str.strip()
    df[desc_output] = df[desc_output].apply(lambda x:re.sub(r'^ac$|^v$|^ac v$|^ac v$|^v ac$|^ac dc$|^dc$|^dc v$|^test$|^part$|^unit$|^part use$|^use part$|^anti$|^micro$|^size$|^engineering$|^material$','empty',x))
    df[desc_output] = df[desc_output].str.replace('\s+', ' ',regex=True)
    df[desc_output] = df[desc_output].apply(lambda x:re.sub(r'^ball screw$','ballscrew',x))
    df[desc_output] = df[desc_output].apply(lambda x: x if len(x) > 3 or x in short_words_keep or unspsc_words else 'empty')
    df[desc_output] = df[desc_output].str.replace(r'^\s*$', 'empty', regex=True)
    df[desc_output] = df[desc_output].str.replace(r'^pendant$', 'robotics', regex=True)
    return df