from os.path import dirname, join

from bokeh.layouts import row, column
from bokeh.plotting import figure, curdoc
from bokeh.models import Circle, Button, ColumnDataSource, CDSView, IndexFilter, ColorBar, LinearColorMapper, LogColorMapper, LogTicker, FixedTicker, FuncTickFormatter
from bokeh.models.widgets import Paragraph, Select, TextInput, AutocompleteInput, CheckboxButtonGroup, PreText
from bokeh.layouts import widgetbox
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import d3
from bokeh.models.callbacks import CustomJS
from bokeh.io import export_svgs
import colorcet as cc
from tabulate import tabulate

import pandas
import numpy as np
from scipy.io import mmread
import pdb;

# Set data dir
data_path = "/large_data/SingleCellData/"
#data_path = "/large_data/SingleCellData/GMM-Demux-input/"

# Call back functions
def selection_func():
    global view
    view.filters = [IndexFilter(source.selected.indices)]


def remove_func():
    global view, source
    remove_set = set(source.selected.indices)
    if len(view.filters) == 0:
        view.filters = [IndexFilter(range(source.data.shape[0]))]
    remain_indices = [x for x in view.filters[0].indices if x not in source.selected.indices]
    view.filters = [IndexFilter(remain_indices)]


def highlight_colored_cells_func(attr, old, new):
    global s_c_c, cur_color, source, s_c
    
    selected_idx_ary = np.argwhere(source.data["color"] == int(s_c_c.value))
    cur_color = int(s_c_c.value)
    
    if len(selected_idx_ary) != 0:
        source.selected.indices = np.concatenate(selected_idx_ary).tolist()
    else:
        source.selected.indices = []

    s_c.value = s_c_c.value
    s_c.css_classes = ["color", "color_" + s_c.value]


def select_color_func(attr, old, new):
    global s_c, cur_color, figures, source

    cur_color = int(s_c.value)

    for figure in figures:
        figure.refresh()

    # Just to trigger plot update 
    source.data["color"] = source.data["color"]


def color_func():
    global source, cur_color, data

    color_list = source.data["color"]

    for i in source.selected.indices:
        color_list[i] = cur_color

    source.data["color"] = color_list
    data["color"] = color_list


def correct_func():
    global source
    source.selected.indices = []


def showall_func():
    global view
    view.filters = []


def tag_func(selector, effector, attr, plot):
    axis = getattr(plot, attr + "axis")
    axis.axis_label = selector.value
    setattr(effector, attr, selector.value)


def toggle_svg_func(active):
    global figures
    print(active)
    for figure in figures:
        if len(active) == 0:
            print("svg turned off!")
            figure.p.output_backend = "canvas"
        if len(active) != 0:
            print("svg turned on!")
            figure.p.output_backend = "svg"
            print("backend is now: ", figure.p.output_backend)
            export_svgs(figure.p, filename="plot.svg")
            print("backend after: ", figure.p.output_backend)


def export_func():
    global source
    _data = source.to_df() #_data = _data.set_index("index")
    print(_data)
    _data.to_csv("bokeh_results.csv")


def exp_func(all_data = False):
    global source, exp_source, exp_input, data
    if all_data:
        exp_data = data
    else:
        exp_data = data.iloc[source.selected.indices]
    exp_source.data = ColumnDataSource.from_df(exp_data)
    exp_input.value = str(int(exp_input.value) + 1)


def reset_func():
    global source, view
    view.filters = []
    source.data["color"] = np.full(len(source.data["color"]), 0)
    source.selected.indices = []


def count_func(attr, old, new):
    global source, view, count_msg, dev_msg
    count_msg.text = "Selected cells: " + str(len(source.selected.indices))

    if dev_msg != None:
        selected_color_df = pandas.DataFrame([[i, 0] for i in source.data["color"][source.selected.indices]], columns=["color", "percentage"])
        selected_color_df = selected_color_df.groupby(['color']).count()
        selected_color_df.loc[:,"percentage"] = selected_color_df.loc[:,"percentage"] / selected_color_df.loc[:,"percentage"].sum()
        dev_msg.text = tabulate(selected_color_df, headers='keys', tablefmt='psql')
        print(dev_msg.text)


def debug_func():
    global figures, err_msg
    err_msg.text = figures[0].p.output_backend


TOOLTIPS = [
        ("index", "@index"),
        ("(x,y)", "($x, $y)"),
        ("color", "@color"),
]

class FlowPlot:
    def __init__(self, opts, source, view, columns, color_map, title = "", x_init_idx = 0, y_init_idx = 0, allow_select = True, select_color_change = True, legend = None):
        self.opts = opts
        self.source = source
        self.view = view
        self.columns = columns
        self.color_map = color_map
        self.p = figure(**self.opts, title = title, tooltips=TOOLTIPS)
        #self.p.output_backend = "svg"
        print("backend is ", self.p.output_backend)
        self.p.xaxis.axis_label = self.columns[x_init_idx]
        self.p.yaxis.axis_label = self.columns[y_init_idx]
        self.r = self.p.circle(self.columns[x_init_idx], self.columns[y_init_idx], source=self.source, view = self.view, line_color=None, fill_color=self.color_map, nonselection_line_color=None, nonselection_fill_color=self.color_map, nonselection_fill_alpha=0.65, legend=legend)
        self.p.legend.click_policy="hide"
        self.s_x = Select(title="x:", value=self.columns[x_init_idx], options=self.columns)
        self.s_y = Select(title="y:", value=self.columns[y_init_idx], options=self.columns)
        # Attach reaction
        self.s_x.on_change("value", lambda attr, old, new: tag_func(self.s_x, self.r.glyph, 'x', self.p) )
        self.s_y.on_change("value", lambda attr, old, new: tag_func(self.s_y, self.r.glyph, 'y', self.p) )
        # Set default fill color
        if select_color_change:
            self.r.selection_glyph = Circle(fill_alpha=1, fill_color=d3['Category20c'][20][0], line_color=None)
        self.allow_select = allow_select

    def refresh(self):
        global cur_color
        self.r.selection_glyph.fill_color = d3['Category20c'][20][cur_color]


    def getColumn(self):
        if self.allow_select:
            return column(self.p, self.s_x, self.s_y)
        else:
            return column(self.p)


    def addColorBar(self, color_bar):
        self.p.add_layout(color_bar, 'right')


# Parsing arguments
args = curdoc().session_context.request.arguments
print(args)

try:
    d_file_names = args.get('file')
    for i in range(len(d_file_names) ):
        d_file_names[i] = d_file_names[i].decode("utf-8") + ".csv"
except:
    d_file_names = ["ADT.csv"]

print("main file names: ", d_file_names)

try:
    num_figs = int(args.get('num')[0].decode("utf-8"))
    print("number of figures: ", num_figs)
except:
    num_figs = 1

try:
    RNA_file = args.get('rna')[0].decode("utf-8")
    print("RNA file is: ", RNA_file)
except:
    RNA_file = None

try:
    tsne_file = args.get('tsne')[0].decode("utf-8") + ".csv"
    print("t-SNE file is: ", tsne_file)
except:
    tsne_file = None

try:
    color_file = args.get('color')[0].decode("utf-8") + ".csv"
    print("color file is: ", color_file)
except:
    color_file = None

try:
    data_dir = args.get('dir')[0].decode("utf-8") + '/'
    print("data_dir is: ", data_dir)
except:
    data_dir = "CITE-1/" 



# Stop if there is any error in loading
loading_error = False

# Read in data and combine them properly
data_array = []
try:
    for d_file in d_file_names:
        print("file path is: ", data_path + data_dir + d_file)
        data_array.append(pandas.read_csv(data_path + data_dir + d_file, index_col = 0) )
except:
    loading_error = True

if not loading_error:
    data = pandas.concat(data_array, axis=1, join='inner')
    generic_columns = data.columns.values.tolist()

# Add tsne data if available
if tsne_file is not None:
    try:
        tsne_data = pandas.read_csv(data_path + data_dir + tsne_file, index_col = 0)
        data = pandas.concat([data, tsne_data], axis=1, join='inner')
    except:
        loading_error = True

# Take color data if available
if color_file is not None:
    try:
        color_data = pandas.read_csv(data_path + data_dir + color_file, index_col = 0)
        color_data.columns.values[0] = "color"
        data = pandas.concat([data, color_data], axis=1, join='inner')
    except:
        print("except in color!")
        loading_error = True

# Stop if error
if not loading_error:

    # Initialize color attribute
    if color_file is None:
        data['color'] = pandas.Series(np.full(data.shape[0], 0), index=data.index)
    data['hl_gene'] = pandas.Series(np.full(data.shape[0], 0), index=data.index)

    # Initialize data source
    opts = dict(plot_width=500, plot_height=500, min_border=0, tools="pan,lasso_select,box_select,wheel_zoom,save")

    source = ColumnDataSource(data)
    view = CDSView(source=source, filters=[IndexFilter([i for i in range(data.shape[0])])])
    color_map = linear_cmap('color', d3['Category20c'][20], low=0, high=20)

    # Configure plots
    figures = []
    for i in range(num_figs):
        figures.append(FlowPlot(opts, source, view, generic_columns, color_map, "Surface Marker Gating Panel") )
    print(figures)

    # Show selection count
    source.selected.on_change("indices", count_func)

    # Gate and color buttons
    g_button = Button(label="Gate")
    g_button.on_click(selection_func)

    rg_button = Button(label="Remove")
    rg_button.on_click(remove_func)

    cur_color = 0

    color_JS_callback = CustomJS(args=dict(color_ary=d3['Category20c'][20]), code="""
            // get data source from Callback args
            var t = $(\".color option:selected\").text();
            $(\".color select\").css(\"color\", color_ary[Number(t)]);
        """)
    s_c = Select(title="Select color:", options=[(i, str(i)) for i in range(20)], value=str(0), css_classes=["color", "color_0"], callback=color_JS_callback)
    s_c.on_change("value", select_color_func)

    c_button = Button(label="Color")
    c_button.on_click(color_func)

    cor_button = Button(label="Correct Plot")
    cor_button.on_click(correct_func)

    s_button = Button(label="Show All")
    s_button.on_click(showall_func)

    pick_color_JS_callback = CustomJS(args=dict(color_ary=d3['Category20c'][20]), code="""
            // get data source from Callback args
            var t = $(\".pick_color option:selected\").text();
            $(\".pick_color select\").css(\"color\", color_ary[Number(t)]);
        """)
    s_c_c = Select(title="Pick colored cells:", options=[(i, str(i)) for i in range(20)], value=str(0), css_classes=["pick_color", "color_0"], callback=pick_color_JS_callback)
    s_c_c.on_change("value", highlight_colored_cells_func)

    r_button = Button(label="Reset Plot")
    r_button.on_click(reset_func)

    svg_button = CheckboxButtonGroup(
        labels=["SVG Plot"], active=[])
    svg_button.on_click(toggle_svg_func)

    #exp_button = Button(label="Export selection")
    #exp_button.on_click(export_func)

    exp_source = ColumnDataSource()

    exp_sel_button = Button(label="Export Selection")
    exp_sel_button.on_click(exp_func)

    exp_all_button = Button(label="Export All Cells")
    exp_all_button.on_click(lambda: exp_func(True))

    exp_input = TextInput(value="0", title="Label:")

    exp_callback = CustomJS(args = dict(source = exp_source, target = exp_input), code=open(join(dirname(__file__), "download.js") ).read() )
    exp_input.js_on_change("value", exp_callback)
    exp_box = widgetbox(exp_input, css_classes = ["hidden"])

    debug_button = Button(label="Debug")
    debug_button.on_click(debug_func)

    err_msg = Paragraph(text="")
    count_msg = Paragraph(text="Selected cells: 0")
    dev_msg = None
    dev_msg = PreText() 

    # Create tsne plot if available
    if tsne_file is not None:
        figures.append(FlowPlot(opts, source, view, data.columns.values.tolist(), color_map, "RNA TSNE Plot", len(generic_columns), len(generic_columns) + 1, False, legend = "color") )

    # Pack figures in columns
    figure_cols = [figure.getColumn() for figure in figures]

    # Package all buttons into a single panel
    #control_panel = [g_button, s_c, row(c_button, cor_button), s_button, r_button, svg_button, exp_button, exp_box, debug_button, err_msg]
    #control_panel = [g_button, rg_button, s_c, row(c_button, cor_button), s_c_c, s_button, r_button, exp_sel_button, exp_all_button, exp_box, count_msg, err_msg]
    control_panel = [g_button, rg_button, s_c, row(c_button, cor_button), s_c_c, s_button, r_button, exp_sel_button, exp_all_button, exp_box, count_msg, err_msg, dev_msg]

    # If also have gene highlight plot
    if RNA_file is not None:
        RNA_error = False

        def hl_gene_func():
            global hl_input, hl_gene_plot, RNA_data, err_msg

            gene_name = hl_input.value
            print(gene_name)
            if gene_name in RNA_data.columns:
                print("in source")
                # Just to trigger plot update 
                #updated_color = ((RNA_data[gene_name] - RNA_data[gene_name].min()) / (RNA_data[gene_name].max() - RNA_data[gene_name].min() ) * 255).apply(np.floor).tolist()
                updated_color = RNA_data[gene_name].tolist()
                source.data["hl_gene"] = updated_color
                #print(source.data["hl_gene"])
            else:
                err_msg.text = "Gene \"" + gene_name + "\" is not in the column. Please check if the gene name matches the autocompletion hint (Letter cases have to match exactly)."


        def read_RNA_matrix(dir_name):
            cell_matrix = (mmread(dir_name + '/matrix.mtx'))
            cell_matrix = cell_matrix.todense()

            cell_names = open(dir_name + "/barcodes.tsv").read().splitlines()
            features = open(dir_name + "/genes.tsv").read().splitlines()

            return pandas.DataFrame(cell_matrix, features, cell_names).T


        print("Adding stuff!!")
        # Read RNA data
        #RNA_data = pandas.read_csv(RNA_file, index_col = 0)
        try:
            RNA_data = read_RNA_matrix(data_path + data_dir + RNA_file)
            RNA_data = RNA_data.reindex(data.index)
        except:
            RNA_error = True

        if not RNA_error:
            # Prepare widgets 
            hl_gene_map = log_cmap('hl_gene', cc.b_linear_blue_5_95_c73[::-1], low=0, high=1000)
            hl_gene_plot = FlowPlot(opts, source, view, data.columns.values.tolist(), hl_gene_map, "Gene Expression Viewing Window", select_color_change = False)
            hl_bar_map = LogColorMapper(palette=cc.b_linear_blue_5_95_c73[::-1], low=0, high=1000)
            hl_gene_ticker = FixedTicker(ticks=[1,10,100,1000])
            hl_color_bar = ColorBar(color_mapper=hl_bar_map, ticker=hl_gene_ticker, label_standoff=8, border_line_color=None, location=(0,0))
            hl_gene_plot.addColorBar(hl_color_bar)

            figure_cols.append(hl_gene_plot.getColumn())
            
            hl_input = AutocompleteInput(completions=RNA_data.columns.values.tolist(), title="Highlight Gene: ")
            hl_button = Button(label="Highlight Gene")
            hl_button.on_click(hl_gene_func)

            control_panel.extend((hl_input, hl_button))

    # This is just for HTO:
    print("***HTO",args.get("HTO"))
    if args.get("HTO") is not None and RNA_file is None:
        def hl_gene_func():
            global hl_input, hl_gene_plot, data, err_msg

            gene_name = hl_input.value
            print(gene_name)
            if gene_name in data.columns:
                print("in source")
                # Just to trigger plot update 
                #updated_color = ((RNA_data[gene_name] - RNA_data[gene_name].min()) / (RNA_data[gene_name].max() - RNA_data[gene_name].min() ) * 255).apply(np.floor).tolist()
                updated_color = data[gene_name].tolist()
                source.data["hl_gene"] = updated_color
                #print(source.data["hl_gene"])
            else:
                err_msg.text = "Gene \"" + gene_name + "\" is not in the column. Please check if the gene name matches the autocompletion hint (Letter cases have to match exactly)."


        hl_gene_map = log_cmap('hl_gene', cc.b_linear_blue_5_95_c73[::-1], low=0, high=1000)
        hl_gene_plot = FlowPlot(opts, source, view, data.columns.values.tolist(), hl_gene_map, "Gene Expression Viewing Window", select_color_change = False)
        hl_bar_map = LogColorMapper(palette=cc.b_linear_blue_5_95_c73[::-1], low=0, high=1000)
        hl_gene_ticker = FixedTicker(ticks=[1,10,100,1000])
        hl_color_bar = ColorBar(color_mapper=hl_bar_map, ticker=hl_gene_ticker, label_standoff=8, border_line_color=None, location=(0,0))
        hl_gene_plot.addColorBar(hl_color_bar)

        figure_cols.append(hl_gene_plot.getColumn())
        
        hl_input = AutocompleteInput(completions=data.columns.values.tolist(), title="Highlight HTO: ")
        hl_button = Button(label="Highlight HTO")
        hl_button.on_click(hl_gene_func)

        control_panel.extend((hl_input, hl_button))

        
    print("control_panel: ", control_panel)

    # Finalize page setup
    curdoc().add_root(row(*figure_cols, column(*control_panel) ) )

else:
    curdoc().add_root(row(Paragraph(text = "Error occurred!! Check url!")) )
