import { WidgetView } from '@jupyter-widgets/base';
import { GeoJSON } from 'leaflet';
import L from '../leaflet';
import { LayerShapes } from '../definitions/leaflet-extend';
import { LeafletControlModel, LeafletControlView } from './Control';
import 'leaflet/dist/leaflet.css';
import '@geoman-io/leaflet-geoman-free';
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css';
export declare class LeafletGeomanDrawControlModel extends LeafletControlModel {
    defaults(): {
        _view_name: string;
        _model_name: string;
        current_mode: null;
        hide_controls: boolean;
        data: never[];
        marker: {};
        circlemarker: {
            pathOptions: {};
        };
        circle: {};
        polyline: {
            pathOptions: {};
        };
        rectangle: {};
        polygon: {
            pathOptions: {};
        };
        text: {};
        edit: boolean;
        drag: boolean;
        remove: boolean;
        cut: boolean;
        rotate: boolean;
        visible: {
            s: boolean;
            ns: boolean;
        };
        newpolygon: {};
        _view_module: string;
        _model_module: string;
        options: string[];
        position: string;
    };
}
export declare class LeafletGeomanDrawControlView extends LeafletControlView {
    feature_group_r: GeoJSON;
    feature_group_b: GeoJSON;
    controlOptions: {
        [option: string]: any;
    };
    initialize(parameters: WidgetView.IInitializeParameters<LeafletControlModel>): void;
    create_obj(): void;
    remove(): void;
    setMode(): void;
    properties_type(layer: L.Layer): "polyline" | "circle" | "polygon" | "circlemarker" | "rectangle" | "marker" | undefined;
    layer_to_json(layer: LayerShapes | L.Layer): any;
    event_to_json(eventName: string, eventLayer: LayerShapes | L.Layer): void;
    data_to_layers(): void;
    layers_to_data(): void;
    handle_message(content: {
        msg: string;
    }): void;
}
//# sourceMappingURL=GeomanDrawControl.d.ts.map