
from bokeh.layouts import row,layout, widgetbox, gridplot,column
from bokeh.models import ColumnDataSource, HoverTool, Legend, WidgetBox
import bokeh.palettes as palettes
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, Title, Span, BoxAnnotation, Range1d,DataRange1d
from bokeh.models.widgets import Div, Paragraph
import pandas as pd
import numpy as np
from config.config import date_asof



# gateways = ['PSC', 'PTC', 'FDJ', 'J1','SoS']
gateways = ['PSC', 'PTC', 'FDJ', 'J1']
brewer = palettes.Set2
rag = palettes.Set1
pdfcolours = brewer[len(gateways)]
ragcolours = rag[4]

pdfcolours = {k:pdfcolours[i] for i,k in enumerate(gateways)}


tools = 'pan,box_zoom,reset,save'

#folder configurations
output_folder ='./output/{}/distribution_all.csv'
cost_folder = './model/data/Cost.csv'
complexity_folder = './model/data/data_complexity.csv'
model_folder = 'data_18_06_2019'#'weibull_baysien'


comp =pd.read_csv(complexity_folder)
cost_dat = pd.read_csv(cost_folder)
cost_dat[['0', '1','2','3','4']] = cost_dat['Program'].str.split(' ', expand = True)

data = pd.read_csv(output_folder.format(model_folder))
data =data[~data.isComplete]
data = data[~ (data.gateway_start =='J1')]
data = data[~ (data.gateway_start =='SoS')]
test_cases_with_predictions = data.test_case[data.isPrediction].unique()
data =data[data.test_case.isin(test_cases_with_predictions)]
tcases = ['L460 2021MY 555e5 R9009','L461 2022MY 545e4 S9010A' ,'L551 2022MY AJ21-P4 235e3 C9162A','X391 2021MY 665e6 J9151',
          'X393 2022MY 544e4 H9211', 'X760 2020MY 335e3 M9023A','L663 2020MY 110 664e6 Y9079A','L560 2021MY I6+PHEV 355e3 T9122A']

data['dates_str'] = data.dates
# data['dates_str'] = data['dates_str'].replace(np.nan, 'NA')
data['survival_curve'] = data['survival_curve'].round(2)
data['pdf_curve'] = data['pdf_curve'].round(5)
data.dates = pd.to_datetime(data.dates,  format = '%Y/%m/%d', exact = True) #TODO check date format
data = data[data.gateway.isin(gateways)]
#
# selected_program = comp['Program Display Name'][comp['total_sum_complexity']>18].values
# data = data[data['test_case'].isin(selected_program)]

output_file(f'dashboard -{model_folder}.html', title = 'Indicative Risk of Programme Delays')

# tcases = data.test_case.unique(
tcases = [i for i in tcases if i in data.test_case.unique()]


sdat = {}
for tc in tcases:
    sdat[tc] = data[data.test_case == tc]


intro_text = f"""
            <p align =right>JLR Corporate and Strategy \ Business Transformation Office \ Analytics CoE</p>            
            <br>
            <br>
            <center><h1>Given delays in current programme gateways, what are the likelihood of subsequent delays?</h1></center>
            <br>
            This visualisation shows the likelihood of various programmes's future gateways' completion dates, given their complexities and the magnitude of current delays as of {date_asof}. 
            The input to the models are delays and complexities in past programmes. <b> The total value at risk for the 5 programmes considered in this dashboard sums up to £974.11 million pounds </b>.
            <br>
            <ul>
            <li>For example, L461 2022MY, the Data Analytics Model (DA) indicates that there is a 98% chance of missing the Cycle Plan timing. We expect J1 to complete on the 1st October 2021, a delay of 133 days compare to cycle plan. 
            There is a 46% chance of missing this date. </li>
            <li>We use Original Baseline and Planned Dates from the Himalaya Reports as of {date_asof} as model inputs.</li>
            <li>The model currently tracks PSC, PTC, FDJ, and J1.</li>
            <li>The values at risk are calculated at DA expected Start of Sales (SoS) dates using the 
            <a href=https://sites.google.com/jaguarlandrover.com/productdevelopmentflow/on-time-value> on-time value</a> provided by the Internal Consulting Division. 
            The calculation uses draft Business Plan volume and margin estimates. It does not consider legislative or regulatory deadlines, strategic importance, 
            resource or prototyping costs.</li>
            <li>We apply Bayesien network and survival analysis to historical gateway delay lengths to compute these likelihood. 
            We have validated against historical J1 dates using delay in PTC or PSC gateways. On average, PTC or PSC gateways are 856 days prior to J1 (minimum of 408 days and maximum of 1233 days).
            The DA model is able to predict J1 with an average absolute error of 73 days and standard deviation of error of 92 days. 54% of the programmes has actual J1 completion dates 
            falling within the 50 percentiles of the DA expected dates. </li>
            <li>We do not consider open issues at various gateways, human resources and supply issues. </li>
            <li>These figures are indicative, rather than predictive. </li>
            """

col1_text = """
            <center><h3> Likelihood of completion dates of remaining gateways</h3></center>
            """


col2_text = """
            <center><h3> Chances of missing various J1 timing</h3></center>
            """

footage_text = """
            <br> 
            <br>
            These figures are indicative, rather than predictive.</i>
            <br>
                        <p align =right><small>Data Reliability Score :7<br>
            Input: 6<br>
            Process :7<br>
            Output :8<br>
            </small>
            </p>
            
            <p align =right><small>Green (9 - 10) : The data is in good shape. You can rely on it to make important decisions both now and in the future.<br>
            Yellow (5 - 8) : The data is OK for making decisions, but you might want to ask some questions about it to make sure it is appropriate for you use case, and you probably don’t want to include it in any long-term processes<br>
            Red (0 - 4) : We strongly recommend that further investigation is done before using this data for any significant business decisions.<br>
            </small>
            </p>
            <br>
            JLR Corporate and Strategy \ Business Transformation Office \ Analytics CoE
            <br>
            <a href=https://mail.google.com/mail/?view=cm&fs=1&to=analytic@jaguarlandrover.com>Email us</a> if there are any queries.
            """

left_col_width = 800
right_col_width = 150
num_right_col = 4
border_width = 100
title_height = 15
intro_div = Div(text =intro_text,width=left_col_width + right_col_width*num_right_col + border_width)
col1_div = Div(text =col1_text,width=left_col_width+ border_width,  height=15)
col2_div = Div(text =col2_text,width=right_col_width*num_right_col, height = 15)
footage_div = Div(text =footage_text,width=left_col_width + right_col_width*num_right_col+ border_width)



# hover tooltip configurations
tooltips = [('gateway', '@gateway'),
            ('date', '@dates_str'),
            ('Chance of missing','@survival_curve{(0:.0%)}')]

def distribution_plot(sdat):
    p = figure(plot_width=left_col_width + border_width, plot_height=300, x_axis_type='datetime', output_backend="webgl")#, tools=tools)
    p.min_border_left = border_width
    p.min_border_top = 0
    legend_wording = []
    max_pdf = 0
    for i, gate in enumerate(gateways):
        colour = pdfcolours[gate]
        sdat_g = sdat[sdat.gateway == gate]

        if sdat_g.shape[0] ==0:
            continue

        sdat_g['0'] = 0

        sdat_week = sdat_g.set_index('dates')[['days' ,'cycle_plan_date' ,'planned_actual_date','expected_date','pdf_curve']].resample('W').sum()
        shape_adjustment = 0.01

        sdat_week['cycle_plan_date'] = sdat_week['cycle_plan_date'].replace(0,np.nan)
        sdat_week['cycle_plan_date'] = sdat_week['cycle_plan_date']-1 -shape_adjustment*(1)

        sdat_week['planned_actual_date'] = sdat_week['planned_actual_date'].replace(0,np.nan)
        sdat_week['planned_actual_date'] = sdat_week['planned_actual_date']-1- shape_adjustment*(2)

        sdat_week['expected_date'] = sdat_week['expected_date'].replace(0,np.nan)
        sdat_week['expected_date'] = sdat_week['expected_date']-1 - shape_adjustment*(3)

        sdat_week['gateway'] = gate
        sdat_week = sdat_week.reset_index()
        sdat_week['dates_str']  =sdat_week.dates.dt.strftime('%d %b %Y')

        sdat_week['survival_curve'] = 1 - sdat_week.pdf_curve.cumsum()
        max_pdf = max(max_pdf, sdat_week.pdf_curve.max())
        if ~sdat_g.isPrediction.unique()[0]:
            sdat_week['pdf_curve'] = np.nan
            sdat_week['survival_curve'] = 0

        sdat_week_vbar = sdat_week[['dates','pdf_curve','gateway','dates_str','survival_curve']]
        source = ColumnDataSource(sdat_week_vbar)
        p.vbar(x='dates', top='pdf_curve', width=0.5, source=source, color=colour, alpha=1)
        liner = p.line(x='dates', y='pdf_curve', source=source, color=colour, alpha=1)

        source = ColumnDataSource(sdat_week)

        circler = p.circle(x='dates', y='planned_actual_date',size =8, source=source, color=colour, alpha=2)
        diamondr = p.diamond(x='dates', y='cycle_plan_date',size =10, source=source, color=colour, alpha=2)
        squarer = p.square(x='dates', y='expected_date',size =8, source=source, color=colour, alpha=2)

        # legend_wording.append((gate,[liner]))
        legend_wording.append((gate,[liner, circler, diamondr, squarer]))

    cutoffdate = BoxAnnotation(right = pd.to_datetime(date_asof,  format = '%Y-%m-%d'),fill_alpha=0.1,fill_color= 'grey')
    cutoffdate_x = pd.to_datetime(date_asof,  format = '%Y-%m-%d') - pd.to_timedelta(4,'M')
    cutoffdate_text = Label(x =cutoffdate_x, y = max_pdf*1.1, text = 'History', render_mode='css', text_alpha = 0.8,text_font_size='9pt')
    p.yaxis.axis_label = 'Probabilities'
    p.xaxis.major_label_orientation = np.pi / 4
    p.add_tools(HoverTool(tooltips= tooltips, mode = 'mouse'))
    p.y_range = Range1d(-0.04, max_pdf*1.4)
    p.x_range = DataRange1d(min(cutoffdate_x - pd.to_timedelta(3,'M'),sdat.dates.min()), sdat.dates.max())
    x_legend = sdat_week.dates.max()-pd.to_timedelta(10,'M')

    vlinecycleplan = Span(location= -0.01, dimension='width', line_color='grey', line_width=0.5, line_alpha = 0.5, line_dash='dashed')
    vlineplanneddates = Span(location=-0.02, dimension='width', line_color='grey', line_width=0.5, line_alpha = 0.5,line_dash='dashed')
    vlineDAexpected = Span(location=-0.03, dimension='width', line_color='grey', line_width=0.5, line_alpha = 0.5,line_dash='dashed')

    cycleplanlegend = Label(x =x_legend, y = -0.01, text = 'Cycle Plan',text_font_size="7pt", border_line_alpha=0, background_fill_alpha=0)
    planneddateslegend = Label(x =x_legend, y = -0.02, text = 'Planned Timing',text_font_size="7pt", border_line_alpha=0, background_fill_alpha=0)
    DAexpectedlegend = Label(x =x_legend, y = -0.03, text = 'DA Expected Date',text_font_size="7pt", border_line_alpha=0, background_fill_alpha=0)
    legend2 = Legend(items =legend_wording, orientation ="vertical")

    p.add_layout(vlinecycleplan)
    p.add_layout(vlineplanneddates)
    p.add_layout(vlineDAexpected)


    p.add_layout(cutoffdate)
    p.add_layout(cutoffdate_text)
    p.add_layout(legend2, "right")
    p.add_layout(cycleplanlegend)
    p.add_layout(planneddateslegend)
    p.add_layout(DAexpectedlegend)
    p.toolbar.logo = None
    p.toolbar_location = None

    title_text = Div(text = '<center><b>Likelihood of completion dates of remaining gateways</b></center>', width=left_col_width + border_width)
    padding_text = Div(text=f'<br><br><br><br><br><br><br>', style={'font-size': '90%'}, width=left_col_width + border_width, height = 50)
    p1 = column(title_text,padding_text, p)
    return(p1)


def VaR(sdat,cost_dat):
    plan = 'expected_date'
    test_case = sdat.test_case.unique()[0]
    ind = [i for i, j in enumerate(cost_dat['0'].tolist()) if j in test_case and cost_dat['0'].iloc[i] in test_case]
    if len(ind)>0:
        cost_per_day = cost_dat['Daily Value of Time million in pounds'].iloc[ind].tolist()[0]
    else:
        cost_per_day = np.nan

    temp_dat = sdat[sdat.gateway == 'J1']
    days = temp_dat[temp_dat[plan] == 1].days.iloc[0]
    workdays = days/7*5

    if ~np.isnan(cost_per_day):
        ValueAtRisk = np.round(workdays*cost_per_day,2)
        return f'DA expected Value at Risk is £{ValueAtRisk} million '
    else:
        return '<br>'



def piechart(sdat,plan, title):
    test_case = sdat.test_case.unique()[0]
    if sdat[sdat.gateway=='J1'].isPrediction.iloc[0]:
        gw = 'J1'
    # else:
    #     gw = 'SoS'
    temp_dat = sdat[sdat.gateway == gw]

    survival= temp_dat[temp_dat[plan] == 1].survival_curve.iloc[0]

    if survival>0.66 and survival<=1:
        rag_col = ragcolours[0]
    elif survival>0.33 and survival<=0.66:
        rag_col = ragcolours[1]
    else:
        rag_col = ragcolours[2]
    datestr = temp_dat[temp_dat[plan]==1]['dates'].dt.strftime('%d %b %y').values[0]
    days = temp_dat[temp_dat[plan] == 1].days.iloc[0]

    p = figure(plot_height = 300,plot_width = right_col_width, x_range =(0.6,1.4))

    p.annular_wedge(x=1,y=-0,inner_radius = 0.3, outer_radius = 0.35, start_angle = 0,
                    end_angle =np.pi*2*survival,color= rag_col)
    if plan == 'planned_actual_date':
        if temp_dat[temp_dat[plan] == 1]['dates'].iloc[0] < pd.to_datetime(date_asof,  format = '%Y-%m-%d'):
            title = 'Actual gateway closed'

    if plan == 'cycle_plan_date':
        delay_days_text = '<br>'

    else:
        delay_days_text = f'{str(int(days))} days delay'



    textg11 = Label(x =0.9,y =-0.06,text=str(int(survival*100))+'%')



    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar_location = None
    p.add_layout(textg11)

    # p.add_layout(Title(text=f'{delay_days_text}',text_font_size="8pt", align="left"), "above")
    # p.add_layout(Title(text=f'{datestr}',text_font_size="8pt", align="left"), "above")
    # p.add_layout(Title(text=f'{title}',text_font_size="9pt", align="left"), "above")
    title_text = Div(text=f'{title}<br>{datestr}<br>{delay_days_text}', style={'font-size': '90%'}, width=right_col_width, height = 50)


    p1 = column(title_text, p)
    return p1



page = [[widgetbox(intro_div)]]

plots_list = [ column(Div(text = f'<b> {tc} </b> <br>{VaR(sdat[tc], cost_dat)} <br><br>', height = 10),
                         row(column(Div(text = '<b><br>Chances of missing J1 timings</b>'),
                             row(
                                  piechart(sdat[tc], 'cycle_plan_date', 'Cycle Plan'),
                                  piechart(sdat[tc], 'planned_actual_date','Planned Timing'),
                                  piechart(sdat[tc], 'expected_date','DA Expected Date'),
                                  piechart(sdat[tc], 'conservative_view', 'DA Conservative View')
                             )
                         ),
                      distribution_plot(sdat[tc])))
               for tc in tcases]
footage = [[widgetbox(footage_div)]]

page.extend(plots_list)
page.extend(footage)
plots = layout(page)
show(plots)
print('test')