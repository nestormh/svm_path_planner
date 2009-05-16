package sibtra.util;

import java.awt.Dimension;
import java.awt.FlowLayout;

import javax.swing.JPanel;

/** Panel con FlowLayout que adapta su tamaño minimo para reflejar el número de 
 * filas que necesita
 * @author alberto
 */
public class PanelFlow extends JPanel {
	
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
//	    Dimension minFL=fL.minimumLayoutSize(this); No produce nada válido
	    int xa=min.width+2*hg; //puntero dentro de la fila actual
	    int ya=min.height+2*vg;
	    for(int i=0; i<getComponentCount();i++) {
	    	Dimension da=getComponent(i).getPreferredSize();
	    	if((xa+da.width)>sizeAct.width) {
	    		//no cabe en la línea actual, hay que saltar de linea
	    		// actulizamos min con los datos de la línea que acaba de terminar
	    		if(xa>min.width) min.width=xa; 
	    		min.height+=ya;
	    		//iniciamos datos de la nueva línea
	    		xa=da.width+hg;
	    		ya=da.height+vg;
	    	} else {
	    		//cabe en la linea
	    		xa+=da.width+hg;
	    		if((da.height+vg)>ya) ya=da.height+vg; //máxima altura
	    	}
	    }
		// actulizamos min con los datos de la última línea
		if(xa>min.width) min.width=xa; 
		min.height+=ya;
	    return min;
	}

}
