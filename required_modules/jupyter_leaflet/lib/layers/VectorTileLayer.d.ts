import { VectorGrid } from 'leaflet';
import L from '../leaflet';
import { LeafletLayerModel, LeafletLayerView } from './Layer';
export declare class LeafletVectorTileLayerModel extends LeafletLayerModel {
    defaults(): {
        _view_name: string;
        _model_name: string;
        url: string;
        vectorTileLayerStyles: {};
        min_zoom: number;
        max_zoom: number;
        min_native_zoom: null;
        max_native_zoom: null;
        interactive: boolean;
        visible: boolean;
        opacity: number;
        rendererFactory: L.TileFactoryFunction<L.SVG.Tile>;
        getFeatureId: null;
        _view_module: string;
        _model_module: string;
        bottom: boolean;
        options: string[];
        name: string;
        base: boolean;
        popup: import("@jupyter-widgets/base").WidgetModel | null;
        popup_min_width: number;
        popup_max_width: number;
        popup_max_height: number | null;
        pane: string;
        subitems: L.Layer[];
    };
}
export declare class LeafletVectorTileLayerView extends LeafletLayerView {
    obj: VectorGrid.Protobuf;
    set_vector_tile_layer_styles(options: any): Promise<any>;
    create_obj(): Promise<void>;
    model_events(): void;
    handle_message(content: {
        msg: string;
    }): void;
}
//# sourceMappingURL=VectorTileLayer.d.ts.map