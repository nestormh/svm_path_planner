package sibtra.util;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Insets;

/** Panel con FlowLayout que adapta su tamaño minimo para reflejar el número de 
 * filas que necesita
 * @author alberto
 */
public class PanelFlow extends PanelDatos {
	
	FlowLayout fL=null;
	
	public PanelFlow() {
		super(new FlowLayout(FlowLayout.LEADING));
		fL=(FlowLayout)getLayout();
	}

	public Dimension getMinimumSize() {
	    Dimension sizePrefe = getPreferredSize();
	    Dimension sizeAct=getSize();
	    //Si cabe bien dejamos tamaño preferido
	    if(sizeAct.width>=sizePrefe.width)
	    	return new Dimension(sizePrefe);
	    Dimension min=new Dimension(0,0);
	    //si no tenemos hijos, 0,0 está bien
	    if(getComponentCount()==0) return min;
	    int vg=fL.getVgap(), hg=fL.getHgap();
		//añadimos los insets por si hay bordes definidos
		Insets ia=getInsets();
//	    Dimension minFL=fL.minimumLayoutSize(this); No produce nada válido
    	Dimension da=getComponent(0).getPreferredSize();
    	//Apuntamos lo de el primer componente
	    int xa=da.width+hg+ia.left+ia.right; //apuntamos desde el principio espacio ocupado por insets
	    int ya=da.height;
	    for(int i=1; i<getComponentCount();i++) {
	    	da=getComponent(i).getPreferredSize();
	    	if((xa+da.width)>sizeAct.width) {
	    		//no cabe en la línea actual, hay que saltar de linea
	    		// actualizamos min con los datos de la línea que acaba de terminar
	    		if(xa>min.width) min.width=xa; 
	    		min.height+=ya+vg;
	    		//iniciamos datos de la nueva línea
	    		xa=da.width+hg+ia.left+ia.right;
	    		ya=da.height;
	    	} else {
	    		//cabe en la linea
	    		xa+=da.width+hg;
	    		if((da.height)>ya) ya=da.height; //máxima altura
	    	}
	    }
		// actulizamos min con los datos de la última línea
		if(xa>min.width) min.width=xa; 
		min.height+=ya+vg+ia.top+ia.bottom; //añadimos espacion necesario para insets
	    return min;
	}

}
