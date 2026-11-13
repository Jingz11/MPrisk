from pathlib import Path
from html import escape
import math

NAVY='#173F59'; TEAL='#298F94'; BLUE='#347CA2'; GOLD='#DCA23B'
MINT='#E5F3F2'; PALEBLUE='#E8F1F6'; PALEGOLD='#FBF1D9'; INK='#000000'

def text(x,y,s,size=25,bold=False,anchor='start',fill=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{fill}">{escape(s)}</text>'
def lines(x,y,ss,size=25,gap=32,anchor='start',bold=False,fill=INK):
    return ''.join(text(x,y+i*gap,s,size,bold,anchor,fill) for i,s in enumerate(ss))
def rect(x,y,w,h,fill,stroke='none',sw=1,r=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def circle(x,y,r,fill,stroke='none',sw=1):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def path(d,fill='none',stroke=NAVY,sw=2,extra=''):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'
def arrow(d,color=NAVY,sw=2.8,dashed=False):
    return path(d,stroke=color,sw=sw,extra=f'marker-end="url(#arrow)"'+(' stroke-dasharray="7 7"' if dashed else ''))
def wrap(body,w,h):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="{NAVY}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
    <rect width="{w}" height="{h}" fill="white"/><g font-family="Arial, Helvetica, sans-serif">{body}</g></svg>'''

def header(x,w,label,num):
    b=path(f'M{x} 28H{x+w-24}L{x+w} 58L{x+w-24} 88H{x}Z',NAVY,'none',0)
    b+=circle(x+32,58,18,'white')+text(x+32,65,str(num),20,True,'middle',INK)
    b+=text(x+64,69,label,29,True,fill='white')
    return b

def landscape(x,y,scale=1):
    # A river–lake cutaway: blue water, explicit banks, flow and granular bed.
    water='#449DCA'; deep='#287BAA'; shore='#52775B'; land='#DCE9D6'
    sand='#D8B57C'; earth='#A87947'; dark='#6C4B2C'
    b=''
    # Solid earth block with a visible front and right sediment face.
    b+=path('M3 57L265 35L298 125L28 153Z',land,shore,1.4)
    b+=path('M28 153L298 125V160L28 186Z',sand,earth,1.5)
    b+=path('M265 35L298 125V160L265 68Z','#B88B57',earth,1.5)
    b+=path('M28 166L298 139V151L28 178Z','#C69B62','none',0)
    # Meandering river broadens toward the foreground. Tributary joins it.
    river='M65 52C45 66 62 77 86 82C118 89 84 104 66 112C51 120 58 136 75 148L111 145C84 132 81 123 104 112C136 96 132 79 99 73C78 69 78 62 93 50Z'
    b+=path(river,water,deep,1.5)
    b+=path('M21 76C36 73 40 93 71 94L82 89C48 90 44 69 17 70Z',water,deep,1)
    # Blue water at the cut face, with sediment beneath.
    b+=path('M75 148L111 145V159Q95 166 75 162Z',deep,deep,1)
    b+=path('M75 162Q93 170 111 159',stroke=dark,sw=1.5)
    # Lake basin: irregular shoreline, shallow fringe and open water.
    lake='M169 76C175 62 193 57 212 59C226 51 249 60 260 72C280 77 285 97 272 111C263 124 240 122 224 131C202 139 178 126 167 113C154 99 157 85 169 76Z'
    b+=path(lake,'#A5CDE1',deep,1.7)
    b+=path('M178 80C187 66 207 69 220 65C240 64 247 78 262 83C277 99 258 111 244 112C229 116 215 128 197 121C180 118 166 106 168 94C168 88 173 84 178 80Z',water,'none',0)
    # A cutaway tongue exposes the lake water column and curved lakebed.
    b+=path('M195 127Q217 139 246 124L251 131V150Q226 163 202 150Z',deep,deep,1)
    b+=path('M202 150Q225 164 251 150',stroke=dark,sw=2)
    # Water texture and directional arrows (generic hydraulic illustration).
    b+=path('M65 63Q67 71 82 75M101 89Q113 96 99 103M74 120Q68 132 85 139',stroke='#D7F0FA',sw=2.4)
    b+=path('M190 88Q204 84 215 88M229 102Q242 98 255 102M181 106Q191 102 202 106',stroke='#D7F0FA',sw=2)
    b+=path('M88 121L86 135M81 129L86 135L91 129',stroke='white',sw=2.3)
    b+=path('M65 66L78 72M72 66L78 72L70 75',stroke='white',sw=2.1)
    # Granular particles in the sediment; deterministic positions, not data.
    for i in range(35):
        px=35+(i*29)%250;py=156+(i*11)%24-(px-28)*.10
        if 33<px<292 and not (75<=px<=111 and py<165) and not (195<=px<=253 and py<153):
            b+=circle(px,py,1.5+(i%3)*.45,[earth,dark,'#E9CF9C'][i%3])
    for px,py in [(40,145),(128,133),(150,129),(278,120)]:
        b+=path(f'M{px} {py}l-3 -7M{px} {py}l2 -9M{px} {py}l6 -5',stroke=shore,sw=1.6)
    # A few riparian tree crowns make the banks immediately recognizable.
    for px,py,rr in [(122,52,7),(144,48,8),(28,62,6),(277,86,6)]:
        b+=path(f'M{px} {py+6}V{py+16}',stroke=earth,sw=2)
        b+=circle(px-3,py,rr,'#729967')+circle(px+3,py-3,rr,'#8AA779')
    return f'<g transform="translate({x},{y}) scale({scale})">{b}</g>'

def particles(x,y,scale=1):
    b=path('M0 32C9 16 26 22 31 6C43 1 50 22 64 24C75 40 54 43 49 60C33 68 29 47 12 47Z',TEAL,'none',0)
    b+=path('M95 8C104 27 98 29 115 40C119 57 93 58 92 70C76 69 75 52 64 43C61 30 83 23 81 10Z',GOLD,'none',0)
    b+=path('M134 19L158 7L181 22L169 47L143 56L129 39Z',BLUE,'none',0)
    b+=path('M201 12C180 27 219 38 198 56C189 67 221 67 228 82',stroke=TEAL,sw=7)
    b+=circle(261,30,15,PALEGOLD,GOLD,3)+circle(275,65,7,PALEBLUE,BLUE,2)
    return f'<g transform="translate({x},{y}) scale({scale})">{b}</g>'

def gear(cx,cy,r,color,pale,label,sub):
    points=[];teeth=10
    for i in range(teeth*4):
        a=-math.pi/2+2*math.pi*i/(teeth*4);rad=r*(1 if i%4 in [1,2] else .90)
        points.append(f'{cx+rad*math.cos(a):.2f},{cy+rad*math.sin(a):.2f}')
    b=f'<polygon points="{" ".join(points)}" fill="{color}"/>'
    b+=circle(cx,cy,r*.73,pale)+circle(cx,cy,r*.64,'none',color,1.2)
    b+=text(cx,cy-4,label,55,True,'middle')
    b+=lines(cx,cy+28,sub,24,28,'middle',True)
    return b

def score(x,y,w=330):
    b=text(x+w/2,y,'Z',82,True,'middle')+text(x+w/2,y+39,'Screening score',29,True,'middle')
    b+=text(x+w/2,y+73,'+ composition uncertainty',24,False,'middle')
    cols=['#E7F2EF','#B9DAD5','#F5DB94','#E7AD63','#D27255']
    for i,c in enumerate(cols):
        b+=rect(x+i*w/5,y+108,w/5-2,26,c)
        b+=text(x+(i+.5)*w/5,y+161,'G'+str(i+1),23,False,'middle')
    b+=path(f'M{x+w*.43} {y+91}V{y+81}H{x+w*.79}V{y+91}',stroke=NAVY,sw=3)
    b+=circle(x+w*.60,y+86,5,NAVY)
    b+=text(x,y+194,'0',22)+text(x+w,y+194,'100',22,anchor='end')
    return b

def network(x,y,s=1):
    p=[(0,10),(0,60),(45,-4),(45,35),(45,78),(90,35)]
    b=''
    for a in [0,1]:
        for c in [2,3,4]:b+=path(f'M{p[a][0]} {p[a][1]}L{p[c][0]} {p[c][1]}',stroke=BLUE,sw=2)
    for a in [2,3,4]:b+=path(f'M{p[a][0]} {p[a][1]}L90 35',stroke=BLUE,sw=2)
    for i,(px,py) in enumerate(p):b+=circle(px,py,8,[TEAL,TEAL,BLUE,BLUE,BLUE,GOLD][i])
    return f'<g transform="translate({x},{y}) scale({s})">{b}</g>'

def association(x,y):
    b=path('M50 0V80',stroke=BLUE,sw=1.4,extra='stroke-dasharray="4 5"')
    for i,(a,c) in enumerate([(23,67),(5,59),(38,94)]):
        yy=10+i*28;b+=path(f'M{a} {yy}H{c}',stroke=TEAL,sw=3)+circle((a+c)/2,yy,5,TEAL)
    return f'<g transform="translate({x},{y})">{b}</g>'

def context_icons(x,y):
    # Recognizable agriculture, industrial infrastructure and urban population.
    green='#607E49'; palegreen='#BDD09D'; soil='#B49059'; blue='#387B9E'
    b=''
    # Perspective crop field with alternating cultivated rows.
    b+=path('M0 59L76 51L113 83L18 95Z',palegreen,green,1.2)
    for j in range(5):
        xx=5+j*18
        b+=path(f'M{xx} {59-j*1.5}L{xx+23} {92-j*2.5}',stroke=soil,sw=5)
    for px,py in [(14,56),(31,54),(48,52)]:
        b+=path(f'M{px} {py}V{py-23}',stroke=green,sw=2)
        b+=path(f'M{px} {py-12}Q{px-12} {py-12} {px-10} {py-23}Q{px} {py-22} {px} {py-12}Z',green,'none',0)
        b+=path(f'M{px} {py-5}Q{px+12} {py-8} {px+10} {py-18}Q{px} {py-16} {px} {py-5}Z',green,'none',0)
    # Tractor: cabin, bonnet and two unequal wheels.
    b+=path('M57 42H69V22H91L97 42H108V56H56Z',GOLD,NAVY,1.3)
    b+=path('M72 25H87L92 41H72Z',PALEBLUE,NAVY,1.2)
    b+=rect(98,34,4,11,NAVY)
    b+=circle(69,55,10,NAVY)+circle(69,55,5,'white')
    b+=circle(101,56,7,NAVY)+circle(101,56,3,'white')
    # Factory with chimneys, toothed roof, windows and a roller door.
    b+=rect(133,28,12,60,blue,NAVY,1.2)+rect(151,14,12,74,blue,NAVY,1.2)
    b+=rect(131,24,16,5,NAVY)+rect(149,10,16,5,NAVY)
    b+=path('M132 60H169V50L188 62V50L207 62V50L226 62V89H132Z',blue,NAVY,1.4)
    for xx in [141,156,174,190,206]:
        b+=rect(xx,69,8,9,'white')
    b+=rect(174,79,23,10,'#A4CCDC',NAVY,1)
    b+=path('M129 91H229',stroke=green,sw=3)
    # City skyline, roofs and regularly spaced windows.
    b+=rect(249,35,26,55,'#85ACC0',NAVY,1.3)
    b+=rect(278,13,29,77,blue,NAVY,1.3)
    b+=rect(310,46,25,44,'#B5CFDC',NAVY,1.3)
    b+=path('M278 13L292 4L307 13Z',NAVY,'none',0)
    for xx,yy,rows in [(255,42,4),(284,22,6),(316,53,3)]:
        for r in range(rows):
            b+=rect(xx,yy+r*11,5,6,'white')+rect(xx+9,yy+r*11,5,6,'white')
    # Adult and child silhouettes denote demographic context.
    b+=circle(349,47,6,NAVY)+circle(367,62,4.5,NAVY)
    b+=path('M344 57Q349 53 354 57L357 73L351 74V92H346L344 74L340 73Z',NAVY,'none',0)
    b+=path('M364 69Q367 66 371 70L373 82L370 83V93H366L364 82L360 78Z',NAVY,'none',0)
    b+=path('M354 62L361 73',stroke=NAVY,sw=4)
    return f'<g transform="translate({x},{y})">{b}</g>'



def render(output):
    b=header(35,414,'Monitoring evidence',1)+header(470,596,'E–C–H framework',2)+header(1087,414,'Screening and use',3)
    b+=path('M456 135V778M1078 135V778',stroke='#D8E5EA',sw=2)
    b+=text(242,132,'River and lake observations',27,True,'middle')
    b+=landscape(75,145,1.10)+text(242,377,'Water · sediment',26,False,'middle')
    b+=particles(85,385,1.05)
    b+=text(242,505,'Abundance and particle profiles',26,True,'middle')
    b+=text(242,542,'Size · shape · colour · polymer',24,False,'middle')
    b+=arrow('M242 559V590')
    b+=text(242,628,'Conditional evidence',28,True,'middle')
    for r in range(6):
        for c in range(16):
            color=[TEAL,BLUE,GOLD][c%3];xx=94+c*19.5;yy=697+r*11
            b+=circle(xx,yy,3.7,color if (r+c)%4 else 'white',color,1.2)
    b+=text(242,790,'96 composition states',25,True,'middle')
    b+=arrow('M424 700C488 700 484 547 542 529')

    # Gears communicate interchangeable modules, without adding a product term.
    b+=gear(770,269,120,TEAL,MINT,'E',['Exposure'])
    b+=gear(647,474,120,GOLD,PALEGOLD,'C',['Contact','opportunity'])
    b+=gear(893,474,120,BLUE,PALEBLUE,'H',['Material','potential'])
    b+=path('M770 363V417M727 475L753 446M813 475L787 446',stroke=NAVY,sw=3)
    b+=circle(770,428,44,'white',NAVY,2)
    b+=text(770,437,'Σ',36,True,'middle')
    b+=text(770,633,'Exposure-anchored',29,True,'middle')
    b+=text(770,671,'dynamic fusion',29,True,'middle')
    b+=path('M564 699H976',stroke=NAVY,sw=2)
    b+=text(770,740,'Fixed references within each medium',25,True,'middle')
    b+=text(770,777,'Water  |  Sediment',25,False,'middle')
    b+=arrow('M1023 426H1110')

    b+=score(1130,221,330)
    b+=path('M1130 451H1460',stroke=NAVY,sw=2)
    b+=text(1295,491,'Targeted monitoring',28,True,'middle')
    b+=rect(1140,534,86,32,PALEGOLD,NAVY,1.8)
    for j in range(8):b+=path(f'M{1147+j*10} 535V{550 if j%3 else 558}',stroke=NAVY,sw=1.5)
    b+=lines(1251,546,['Information','value'],27,32)
    for x,y in [(1145,659),(1182,644),(1217,667)]:b+=circle(x,y,9,TEAL)
    b+=arrow('M1170 687H1224')
    b+=lines(1251,665,['Local','updating'],27,32)
    # Context bypasses the score; it enters downstream analyses only.
    b+=rect(35,837,1466,107,PALEBLUE,r=4)
    b+=context_icons(60,844)+lines(455,876,['Environmental and','socioeconomic context'],23,30)
    b+=arrow('M759 891H824')
    b+=lines(870,876,['External prediction','and pressure associations'],23,30)
    (output/'Figure_1.svg').write_text(wrap(b,1536,975),'utf8')
