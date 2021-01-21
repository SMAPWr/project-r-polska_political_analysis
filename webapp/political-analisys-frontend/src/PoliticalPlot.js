import React from 'react';

import Plotly from "plotly.js"
import createPlotlyComponent from 'react-plotly.js/factory';
const Plot = createPlotlyComponent(Plotly);

export default function PoliticalPlot(props) {
    return (
        <Plot
            data={[
                {
                    x: [props.economic],
                    y: [props.worldview],
                    type: 'scatter',
                    mode: 'markers',
                    marker: { color: 'red' }
                },

            ]}

            layout={{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                width: 400,
                height: 400,
                xaxis: {
                    range: [-1.2, 1.2],
                    title: "economic",
                    showticklabels: false
                },
                yaxis: {
                    range: [-1.2, 1.2],
                    title: "worldview",
                    showticklabels: false
                },
                showlegend: false,
                annotations: [
                    {
                        x: 0,
                        y: 1.15,
                        mode: 'text',
                        text: 'Conservative',
                        showarrow: false,
                        bgcolor: '#ffffff',
                        opacity: 1
                    },
                    {
                        x: 0,
                        y: -1.15,
                        mode: 'text',
                        text: 'Liberal',
                        showarrow: false,
                        bgcolor: '#ffffff',
                        opacity: 1
                    },
                    {
                        x: -1.15,
                        y: 0,
                        mode: 'text',
                        text: 'left',
                        showarrow: false,
                        textangle: '-90',
                        bgcolor: '#ffffff',
                        opacity: 1
                    },
                    {
                        x: 1.15,
                        y: 0,
                        mode: 'text',
                        text: 'right',
                        showarrow: false,
                        textangle: '90',
                        bgcolor: '#ffffff',
                        opacity: 1
                    }]
            }}
            config={{
                displayModeBar: false, 
            }}
        />
    );
}